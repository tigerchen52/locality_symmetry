from classify import main


def finetune(
        model_file='model/finetune/model_steps_1024.pt',
        task='stsb',
        train_cfg='config/common_small.json',
        pretrain_file='model/pretrain/model_steps_40000.pt',
        train_data_file='glue/glue_data/STS-B/train.tsv',
        dev_data_file='glue/glue_data/STS-B/dev.tsv'
):
    score_list = []
    for lr in [3e-4, 1e-4, 5e-5, 3e-5, 1e-5]:
        main(
            task=task,
            train_cfg=train_cfg,
            pretrain_file=pretrain_file,
            data_file=train_data_file,
            mode='train',
            lr=lr
        )

        score = main(
            task=task,
            train_cfg=train_cfg,
            model_file=model_file,
            data_file=dev_data_file,
            mode='eval'
        )

        print('finished, final score = {a}'.format(a=score))
        score_list.append(score)

    print(max(score_list), score_list)
    return max(score_list), score_list


def overall():
    tasks = [
        {
            'task': 'mrpc',
            'train_cfg': 'config/common_small.json',
            'train_data_file': 'glue/glue_data/MRPC/msr_paraphrase_train.txt',
            'dev_data_file': 'glue/glue_data/MRPC/msr_paraphrase_test.txt'
        },
        {
            'task': 'stsb',
            'train_cfg':'config/common_small.json',
            'train_data_file':'glue/glue_data/STS-B/train.tsv',
            'dev_data_file':'glue/glue_data/STS-B/dev.tsv'
         },
        {
            'task': 'sst2',
            'train_cfg': 'config/common_big.json',
            'train_data_file': 'glue/glue_data/SST-2/train.tsv',
            'dev_data_file': 'glue/glue_data/SST-2/dev.tsv'
        },
        {
            'task': 'qnli',
            'train_cfg': 'config/common_big.json',
            'train_data_file': 'glue/glue_data/QNLI/train.tsv',
            'dev_data_file': 'glue/glue_data/QNLI/dev.tsv'
        },
        {
            'task': 'qqp',
            'train_cfg': 'config/train_mrpc.json',
            'train_data_file': 'glue/glue_data/QQP/train.tsv',
            'dev_data_file': 'glue/glue_data/QQP/dev.tsv'
        },
        {
            'task': 'mnli',
            'train_cfg': 'config/train_mrpc.json',
            'train_data_file': 'glue/glue_data/MNLI/train.tsv',
            'dev_data_file': 'glue/glue_data/MNLI/dev_matched.tsv'
        },
        {
            'task': 'mnli',
            'train_cfg': 'config/train_mrpc.json',
            'train_data_file': 'glue/glue_data/MNLI/train.tsv',
            'dev_data_file': 'glue/glue_data/MNLI/dev_mismatched.tsv'
        }
    ]

    w_l = ''
    pretrain_file = 'model/pretrain/model_finetuned.pt'
    for task in tasks:
        max_score, scores = finetune(
            pretrain_file=pretrain_file,
            task=task['task'],
            train_cfg=task['train_cfg'],
            train_data_file=task['train_data_file'],
            dev_data_file=task['dev_data_file'],
        )
        w_l += task['task'] + '\t' + str(max_score) + '\t' + str(scores) + '\n'
        print(w_l)

    print('finished!')
    print(w_l)


if __name__ == '__main__':
    overall()
