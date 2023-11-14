import pandas as pd


def format_test_metrics(pl_results):
    if type(pl_results) == dict:
        metric_df = pd.DataFrame(pl_results, index=[0])
    elif type(pl_results) == list:
        metric_df = pd.DataFrame(pl_results, index=[i for i in range(len(pl_results))])
        if len(pl_results) != 1:
            print('Please check it! pl_results is a list with more than 1 element.')
    else:
        metric_df = pd.DataFrame(pl_results, index=[0])
    return metric_df


class BinaryClassificationRecorder:
    def __init__(self):
        super(BinaryClassificationRecorder).__init__()
        self.epoch_records = {'valid': [], 'test': []}
        self.sheet_names = {'valid': 'valid_epoch_records', 'test': 'test_epoch_records'}
        self.key_list = ['ACC', 'AUC', 'MCC', 'Q-value', 'F1', 'F0.5', 'F2',
                         'SE', 'SP', 'PPV', 'NPV', 'TN', 'FP', 'FN', 'TP']
        self.record_keys = ['epoch', 'step'] + self.key_list

    def record(self, lit_model, metric_dict, stage):
        path_record = lit_model.logger.log_dir + f'/{self.sheet_names[stage]}.xlsx'
        records = [lit_model.current_epoch, lit_model.global_step] + \
                  [metric_dict[stage + '/' + key + '_epoch'].item() for key in self.key_list]
        self.epoch_records[stage].append(records)
        record_df = pd.DataFrame(
            self.epoch_records[stage],
            index=[i for i in range(len(self.epoch_records[stage]))],
            columns=self.record_keys
        )
        with pd.ExcelWriter(path_record, engine='xlsxwriter') as writer:
            workbook = writer.book
            general_format = workbook.add_format(lit_model.args.xlsx_output_format.general_format)
            num_format = workbook.add_format(lit_model.args.xlsx_output_format.number_format)
            record_df.to_excel(writer, sheet_name=self.sheet_names[stage])
            worksheet = writer.sheets[self.sheet_names[stage]]
            worksheet.set_column('A:R', 16, cell_format=general_format)
            worksheet.set_column('D:N', 16, cell_format=num_format)
