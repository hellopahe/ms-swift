# Copyright (c) Alibaba, Inc. and its affiliates.
import os

from swift.llm import ExportArguments, HfConfigFactory, prepare_model_template, save_checkpoint
from swift.tuners import Swift
from swift.utils import get_logger

logger = get_logger()


def check_tie_word_embeddings(model):
    config = model.config
    try:
        from peft.utils import ModulesToSaveWrapper
        if not HfConfigFactory.get_config_attr(config, 'tie_word_embeddings'):
            return
        for module in [model.get_input_embeddings(), model.get_output_embeddings()]:
            if not isinstance(module, ModulesToSaveWrapper):
                return
        HfConfigFactory.set_config_attr(config, 'tie_word_embeddings', False)
    except Exception:
        pass


def merge_lora(args: ExportArguments, device_map=None, replace_if_exists=False) -> None:
    if replace_if_exists:
        logger.info(f'replace_if_exists: {replace_if_exists}')
    output_dir = getattr(args, 'output_dir', None) or f'{args.adapters[0]}-merged'
    if os.path.exists(output_dir) and not replace_if_exists:
        logger.info(f'The weight directory for the merged LoRA already exists in {output_dir}, '
                    'skipping the saving process.')
    else:
        # If the model is quantized, perform the merge on the original (unquantized) model.
        # https://github.com/huggingface/peft/issues/2321
        args.quant_method = None
        origin_device_map = args.device_map
        args.device_map = device_map or args.device_map
        logger.info(f'merge_device_map: {device_map}')
        model, template = prepare_model_template(args)
        logger.info('Merge LoRA...')
        check_tie_word_embeddings(model)
        
        logger.info(f'[DEBUG] Before merge - model type: {type(model).__name__}')
        if hasattr(model, 'model'):
            logger.info(f'[DEBUG] Before merge - model.model type: {type(model.model).__name__}')
        
        from swift.llm.model.patcher import get_lm_head_model
        llm_model_before = None
        if hasattr(model, 'model') and hasattr(model.model, 'model_meta'):
            llm_model_before = get_lm_head_model(model.model, model.model.model_meta, ['lm_head', 'output', 'embed_out', 'output_layer'])
            logger.info(f'[DEBUG] Before merge - llm_model has score: {hasattr(llm_model_before, "score")}')
        
        Swift.merge_and_unload(model)
        
        logger.info(f'[DEBUG] After merge - model type: {type(model).__name__}')
        if hasattr(model, 'model'):
            logger.info(f'[DEBUG] After merge - model.model type: {type(model.model).__name__}')
            if hasattr(model.model, 'model_meta'):
                llm_model_after = get_lm_head_model(model.model, model.model.model_meta, ['lm_head', 'output', 'embed_out', 'output_layer'])
                logger.info(f'[DEBUG] After merge - llm_model has score: {hasattr(llm_model_after, "score")}')
        
        model = model.model
        logger.info('Saving merged weights...')

        save_checkpoint(
            model,
            template.processor,
            output_dir,
            safe_serialization=args.safe_serialization,
            model_dirs=args.adapters,
            max_shard_size=args.max_shard_size,
            additional_saved_files=model.model_meta.additional_saved_files)
        
        if args.task_type == 'seq_cls' and args.num_labels is not None:
            import json
            config_path = os.path.join(output_dir, 'config.json')
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                config['num_labels'] = args.num_labels
                with open(config_path, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2, ensure_ascii=False)
                logger.info(f'Updated config.json with num_labels={args.num_labels} for seq_cls task')
        
        logger.info(f'Successfully merged LoRA and saved in {output_dir}.')
        args.device_map = origin_device_map

    args.model = output_dir
    args.model_dir = output_dir
    args.adapters = []
