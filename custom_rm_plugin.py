# custom_rm_plugin.py
from typing import List
import torch
from swift.llm import PtEngine, InferRequest
from swift.plugin import ORM, orms
from swift.utils import get_logger

logger = get_logger()


class LocalRMReward(ORM):
    """
    使用本地 RM 模型（PtEngine）作为奖励函数
    基于 examples/infer/demo_reward_model.py 的推理方式
    """
    
    def __init__(self):
        rm_model_path = '/root/autodl-tmp/ckpts/intern_vl3_14b-lora-rm-20Oct2025-1920-ckpt/v0-20251020-192110/checkpoint-30-merged'
        
        logger.info(f'[LocalRM] Loading reward model from: {rm_model_path}')
        
        # 使用 PtEngine 加载 RM，与 demo_reward_model.py 相同的方式
        self.engine = PtEngine(rm_model_path, max_batch_size=64)
        
        logger.info(f'[LocalRM] ✅ Reward model loaded successfully!')
        logger.info(f'[LocalRM] Model type: {type(self.engine.model).__name__}')
    
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
            logger.warning('[LocalRM] No messages provided, returning zero rewards')
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
                logger.warning(f'[LocalRM] Sample {idx}: messages 最后一条不是 assistant，使用 completion 补充')
                if messages_copy and idx < len(completions):
                    messages_copy.append({
                        'role': 'assistant',
                        'content': completions[idx]
                    })
                else:
                    logger.error(f'[LocalRM] Sample {idx}: messages 无效，使用默认分数 0.0')
                    continue
            
            # 创建 InferRequest（包含图片信息）
            request_kwargs = {'messages': messages_copy}
            if images and idx < len(images) and images[idx]:
                request_kwargs['images'] = images[idx]
            rm_requests.append(InferRequest(**request_kwargs))
            valid_indices.append(idx)
        
        # 初始化所有样本的 rewards 为 0.0
        rewards = [0.0] * len(completions)
        
        # 批量推理
        try:
            # 使用 engine.infer 进行批量评分
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
                    logger.warning(f'[LocalRM] Failed to parse reward for sample {original_idx}: {e}, using 0.0')
                    rewards[original_idx] = 0.0
            
            # 记录统计信息
            if rewards:
                avg_reward = sum(rewards) / len(rewards)
                logger.info(f'[LocalRM] Batch size: {len(rewards)}, Avg reward: {avg_reward:.4f}, '
                           f'Min: {min(rewards):.4f}, Max: {max(rewards):.4f}')
            
            return rewards
            
        except Exception as e:
            logger.error(f'[LocalRM] Error during RM inference: {e}')
            # 返回默认分数避免训练中断
            return [0.0] * len(completions)


# 注册自定义奖励函数
orms['local_rm_reward'] = LocalRMReward

logger.info('[LocalRM] ✅ Registered custom reward function: local_rm_reward')

