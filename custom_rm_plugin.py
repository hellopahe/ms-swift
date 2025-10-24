# custom_rm_plugin.py
import re
from typing import List
from swift.llm import InferClient, InferRequest
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()

# 正则表达式匹配思考和回答部分
_WORK_RE = re.compile(
    r"<start_thinking>(.*?)</end_thinking>", flags=re.S | re.I
)


def _strip_reasoning(text: str) -> str:
    """移除 thinking 段并抽取 <start_response> 内文本。"""
    end_pos = text.rfind('</end_response>')
    if end_pos == -1:
        return text.strip()
    
    start_pos = text.rfind('<start_response>', 0, end_pos)
    if start_pos == -1:
        return text.strip()
    
    return text[start_pos + len('<start_response>'):end_pos].strip()


def _extract_thinking(text: str) -> str:
    """提取 thinking 部分的文本"""
    m = _WORK_RE.search(text)
    return (m.group(1) if m else "").strip()


class LocalRMReward(ORM):
    """
    使用本地 RM API 服务作为奖励函数
    通过 HTTP API 调用独立部署的 Reward Model 服务
    
    环境变量配置:
        RM_HOST: RM 服务器地址
                - 支持纯主机名: 127.0.0.1 或 example.com
                - 支持完整URL: https://example.com 或 http://example.com
                默认: 127.0.0.1
        RM_PORT: RM 服务器端口（默认 8001）
                 如果 RM_HOST 包含端口，URL中的端口优先
    """
    
    def __init__(self, rm_host=None, rm_port=None):
        import os
        from urllib.parse import urlparse
        
        # 优先使用传入参数，其次使用环境变量，最后使用默认值
        rm_host_raw = rm_host or os.getenv('RM_HOST', '127.0.0.1')
        rm_port_raw = rm_port or os.getenv('RM_PORT', '8001')
        
        # 解析 URL：如果包含协议（http://或https://），则提取各部分
        if rm_host_raw.startswith('http://') or rm_host_raw.startswith('https://'):
            parsed = urlparse(rm_host_raw)
            scheme = parsed.scheme
            hostname = parsed.hostname
            # URL 中的端口优先于环境变量的端口
            port = parsed.port or int(rm_port_raw)
            base_url = f'{scheme}://{hostname}:{port}/v1'
            
            logger.info(f'[LocalRM API] Connecting to RM server at {base_url}')
            self.engine = InferClient(base_url=base_url)
        else:
            # 兼容旧的纯主机名方式
            self.rm_host = rm_host_raw
            self.rm_port = int(rm_port_raw)
            logger.info(f'[LocalRM API] Connecting to RM server at {self.rm_host}:{self.rm_port}')
            self.engine = InferClient(host=self.rm_host, port=self.rm_port)
        
        # 测试连接
        try:
            models = self.engine.models
            logger.info(f'[LocalRM API] ✅ Connected to RM server successfully!')
            logger.info(f'[LocalRM API] Available models: {models}')
        except Exception as e:
            logger.error(f'[LocalRM API] ❌ Failed to connect to RM server: {e}')
            raise
    
    def __call__(self, completions: List[str], messages=None, images=None, **kwargs) -> List[float]:
        """
        计算奖励分数
        
        注意：会自动去除思考过程，只评估纯答案部分
        
        Args:
            completions: 模型生成的回答列表（已经在 messages 中）
            messages: List[List[Dict]], 每个样本的完整对话历史（包含 assistant 的回答）
            images: List[List], 每个样本的图片列表（可选）
            kwargs: 其他参数
        
        Returns:
            rewards: 奖励分数列表（float）
        """
        if not messages:
            logger.warning('[LocalRM API] No messages provided, returning zero rewards')
            return [0.0] * len(completions)
        
        # 准备 RM 推理请求
        rm_requests = []
        valid_indices = []  # 记录有效样本的索引
        
        for idx, msg_list in enumerate(messages):
            # 复制对话历史（避免修改原数据）
            messages_copy = [msg.copy() for msg in msg_list] if isinstance(msg_list, list) else []
            
            # messages 中最后一条应该已经包含了 assistant 的回答
            # 但为了安全起见，检查并确保最后一条是 assistant 的回答
            if not messages_copy or messages_copy[-1]['role'] != 'assistant':
                logger.warning(f'[LocalRM API] Sample {idx}: messages 最后一条不是 assistant，使用 completion 补充')
                if messages_copy and idx < len(completions):
                    messages_copy.append({
                        'role': 'assistant',
                        'content': completions[idx]
                    })
                else:
                    logger.error(f'[LocalRM API] Sample {idx}: messages 无效，使用默认分数 0.0')
                    continue
            
            # 关键修改：去除思考过程，只保留纯答案
            original_content = messages_copy[-1]['content']
            if isinstance(original_content, str):
                stripped_content = _strip_reasoning(original_content)
                messages_copy[-1] = {
                    'role': 'assistant',
                    'content': stripped_content
                }
                
                # 打印第一个样本的详细内容，方便检查
                if idx == 0:
                    logger.info(f'[LocalRM API] Sample {idx} - Original length: {len(original_content)}, '
                               f'Stripped length: {len(stripped_content)}')
                    logger.info(f'[LocalRM API] Sample {idx} - Original content:\n{original_content}')
                    logger.info(f'[LocalRM API] Sample {idx} - Stripped content (sent to RM):\n{stripped_content}')
                    logger.info(f'>>>>> requested messages: {messages_copy}')
                else:
                    # 其他样本只打印摘要
                    logger.debug(f'[LocalRM API] Sample {idx} - Original: {len(original_content)} chars, '
                                f'Stripped: {len(stripped_content)} chars')
            
            # 创建 InferRequest（包含图片信息）
            request_kwargs = {'messages': messages_copy}
            if images and idx < len(images) and images[idx]:
                request_kwargs['images'] = images[idx]
            rm_requests.append(InferRequest(**request_kwargs))
            valid_indices.append(idx)
        
        # 初始化所有样本的 rewards 为 0.0
        rewards = [0.0] * len(completions)
        
        # 批量推理（调用 API）
        try:
            # 使用 InferClient 批量调用 RM API
            results = self.engine.infer(rm_requests, use_tqdm=False)
            
            for req_idx, result in enumerate(results):
                original_idx = valid_indices[req_idx]  # 映射回原始索引
                try:
                    # RM 返回的分数在 message.content 中
                    score_str = result.choices[0].message.content
                    
                    # 转换为 float
                    if isinstance(score_str, str):
                        score = float(score_str.strip())
                    else:
                        score = float(score_str)
                    
                    rewards[original_idx] = score
                    
                except (ValueError, AttributeError, IndexError) as e:
                    logger.warning(f'[LocalRM API] Failed to parse reward for sample {original_idx}: {e}, using 0.0')
                    rewards[original_idx] = 0.0
            
            # 记录统计信息
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                logger.info(f'[LocalRM API] Batch size: {len(rewards)}, Avg reward: {avg_reward:.4f}, '
                           f'Min: {min(rewards):.4f}, Max: {max(rewards):.4f}')
            
            return rewards
            
        except Exception as e:
            logger.error(f'[LocalRM API] Error during RM API call: {e}')
            logger.error(f'[LocalRM API] Please check if RM server is running')
            # 返回默认分数避免训练中断
            return [0.0] * len(completions)


class ThinkingFormatReward(ORM):
    """
    检查模型输出是否符合思考格式的奖励函数
    
    期望格式：
        <start_thinking>思考内容...</end_thinking>
        <start_response>回答内容...</end_response>
    
    奖励规则：
        - 完全符合格式: 1.0
        - 只有部分标签: 0.5
        - 标签顺序错误: 0.3
        - 完全不符合: 0.0
    """
    
    def __init__(self):
        logger.info('[ThinkingFormat] ✅ Initialized ThinkingFormatReward')
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        检查每个 completion 是否符合思考格式
        
        Args:
            completions: 模型生成的回答列表
            kwargs: 其他参数（不使用）
        
        Returns:
            rewards: 格式奖励分数列表（0.0-1.0）
        """
        rewards = []
        
        for completion in completions:
            # 处理可能的 token_ids 格式
            if isinstance(completion, list):
                # 如果是 token IDs，跳过检查
                logger.warning('[ThinkingFormat] Received token IDs instead of string, skipping')
                rewards.append(0.0)
                continue
            elif isinstance(completion, dict):
                completion = completion.get('content', '')
            
            content = str(completion)
            
            # 去掉 <|im_start|>assistant 及之前的部分
            assistant_pos = content.rfind('<|im_start|>assistant')
            if assistant_pos != -1:
                content = content[assistant_pos + len('<|im_start|>assistant'):]
            
            # 检查是否包含各个标签
            has_thinking_start = '<start_thinking>' in content
            has_thinking_end = '</end_thinking>' in content
            has_response_start = '<start_response>' in content
            has_response_end = '</end_response>' in content
            
            # 检查标签的位置
            thinking_start_pos = content.find('<start_thinking>')
            thinking_end_pos = content.find('</end_thinking>')
            response_start_pos = content.find('<start_response>')
            response_end_pos = content.find('</end_response>')
            
            # 计算奖励
            reward = 0.0
            
            # 完全符合格式
            if (has_thinking_start and has_thinking_end and 
                has_response_start and has_response_end):
                
                # 检查顺序是否正确
                if (thinking_start_pos < thinking_end_pos < 
                    response_start_pos < response_end_pos):
                    reward = 1.0  # 完美格式
                else:
                    reward = 0.3  # 有所有标签但顺序错误
            
            # 部分符合格式
            elif ((has_thinking_start and has_thinking_end) or 
                  (has_response_start and has_response_end)):
                reward = 0.5  # 只有一对标签
            
            # 有部分标签
            elif (has_thinking_start or has_thinking_end or 
                  has_response_start or has_response_end):
                reward = 0.2  # 有个别标签
            
            # 完全不符合
            else:
                reward = 0.0
            
            # 检查是否以 <start_thinking>嗯， 开始（支持全角和半角逗号）
            if has_thinking_start:
                thinking_content_start = content[thinking_start_pos + len('<start_thinking>'):]
                if thinking_content_start.startswith('嗯，') or thinking_content_start.startswith('嗯,'):
                    old_reward = reward
                    reward += 0.5
                    logger.info(f'[ThinkingFormat] ✅ Found "嗯，" at start, reward: {old_reward:.2f} → {reward:.2f}')
            
            rewards.append(reward)
        
        # 记录统计信息
        if rewards:
            avg_reward = sum(rewards) / len(rewards)
            perfect_count = sum(1 for r in rewards if r == 1.0)
            logger.info(
                f'[ThinkingFormat] Batch: {len(rewards)}, '
                f'Avg: {avg_reward:.2f}, Perfect: {perfect_count}/{len(rewards)}'
            )
        
        return rewards


# 注册自定义奖励函数
orms['local_rm_reward'] = LocalRMReward
orms['thinking_format'] = ThinkingFormatReward

logger.info('[LocalRM API] ✅ Registered custom reward function: local_rm_reward')
logger.info('[ThinkingFormat] ✅ Registered custom reward function: thinking_format')

