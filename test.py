import numpy as np
test=np.load('./outputs/gBR_sBM_c01_d05_mBR0_ch02_mBR0.npy')
print(test[0])

configs['model'] = pipeline_config.multi_modal_model
configs['train_config'] = pipeline_config.train_config
configs['train_dataset'] = pipeline_config.train_dataset
configs['eval_config'] = pipeline_config.eval_config
configs['eval_dataset'] = pipeline_config.eval_dataset


 model_config = configs['model']
train_config = configs['train_config']

def dataset_fn(input_context=None):
    del input_context
    train_config = configs['train_config']
    train_dataset_config = configs['train_dataset']
    use_tpu = (FLAGS.train_strategy == TRAIN_STRATEGY[0])
    dataset = inputs.create_input(
        train_config, train_dataset_config, use_tpu=use_tpu)
    return dataset