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
    
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        """
        计算奖励分数
        
        Args:
            completions: 模型生成的回答列表
            kwargs: 包含 'inputs' 等信息
                - inputs: List[Dict], 每个包含 'messages' 等字段
        
        Returns:
            rewards: 奖励分数列表（float）
        """
        inputs = kwargs.get('inputs', [])
        
        if not inputs:
            logger.warning('[LocalRM] No inputs provided, returning zero rewards')
            return [0.0] * len(completions)
        
        # 准备 RM 推理请求
        rm_requests = []
        for inp, completion in zip(inputs, completions):
            # 获取原始对话历史
            messages = inp.get('messages', []).copy()
            
            # 添加模型生成的回答
            if messages and messages[-1]['role'] == 'assistant':
                # 如果最后一条已经是 assistant，替换内容
                messages[-1]['content'] = completion
            else:
                # 否则添加新的 assistant 消息
                messages.append({
                    'role': 'assistant',
                    'content': completion
                })
            
            # 创建 InferRequest
            rm_requests.append(InferRequest(messages=messages))
        
        # 批量推理
        try:
            # 使用 engine.infer 进行批量评分
            results = self.engine.infer(rm_requests, use_tqdm=False)
            
            rewards = []
            for idx, result in enumerate(results):
                try:
                    # RM 返回的分数在 message.content 中
                    score_str = result.choices[0].message.content
                    
                    # 转换为 float
                    if isinstance(score_str, str):
                        score = float(score_str.strip())
                    else:
                        score = float(score_str)
                    
                    rewards.append(score)
                    
                except (ValueError, AttributeError, IndexError) as e:
                    logger.warning(f'[LocalRM] Failed to parse reward for sample {idx}: {e}, using 0.0')
                    rewards.append(0.0)
            
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

