# custom_rm_plugin.py
from typing import List
from swift.llm import InferClient, InferRequest
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class LocalRMReward(ORM):
    """
    使用本地 RM API 服务作为奖励函数
    通过 HTTP API 调用独立部署的 Reward Model 服务
    
    环境变量配置:
        RM_HOST: RM 服务器地址（默认 127.0.0.1）
        RM_PORT: RM 服务器端口（默认 8001）
    """
    
    def __init__(self, rm_host=None, rm_port=None):
        import os
        
        # 优先使用传入参数，其次使用环境变量，最后使用默认值
        self.rm_host = rm_host or os.getenv('RM_HOST', '127.0.0.1')
        self.rm_port = int(rm_port or os.getenv('RM_PORT', '8001'))
        
        logger.info(f'[LocalRM API] Connecting to RM server at {self.rm_host}:{self.rm_port}')
        
        # 使用 InferClient 连接 RM API 服务
        try:
            self.engine = InferClient(host=self.rm_host, port=self.rm_port)
            models = self.engine.models
            logger.info(f'[LocalRM API] ✅ Connected to RM server successfully!')
            logger.info(f'[LocalRM API] Available models: {models}')
        except Exception as e:
            logger.error(f'[LocalRM API] ❌ Failed to connect to RM server: {e}')
            raise
    
    def __call__(self, completions: List[str], messages=None, images=None, **kwargs) -> List[float]:
        """
        计算奖励分数
        
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
            messages_copy = msg_list.copy() if isinstance(msg_list, list) else []
            
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
            logger.error(f'[LocalRM API] Please check if RM server at {self.rm_host}:{self.rm_port} is running')
            # 返回默认分数避免训练中断
            return [0.0] * len(completions)


# 注册自定义奖励函数
orms['local_rm_reward'] = LocalRMReward

logger.info('[LocalRM API] ✅ Registered custom reward function: local_rm_reward')

