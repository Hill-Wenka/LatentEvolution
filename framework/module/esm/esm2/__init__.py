import os

esm_model_names = ['esm1_t34_670M_UR50S',
                   'esm1_t34_670M_UR50D',
                   'esm1_t34_670M_UR100',
                   'esm1_t12_85M_UR50S',
                   'esm1_t6_43M_UR50S',
                   'esm1b_t33_650M_UR50S',
                   'esm_msa1_t12_100M_UR50S',
                   'esm_msa1b_t12_100M_UR50S',
                   'esm1v_t33_650M_UR90S_1',
                   'esm1v_t33_650M_UR90S_2',
                   'esm1v_t33_650M_UR90S_3',
                   'esm1v_t33_650M_UR90S_4',
                   'esm1v_t33_650M_UR90S_5',
                   'esm_if1_gvp4_t16_142M_UR50',
                   'esm2_t6_8M_UR50D',
                   'esm2_t12_35M_UR50D',
                   'esm2_t30_150M_UR50D',
                   'esm2_t33_650M_UR50D',
                   'esm2_t36_3B_UR50D',
                   'esm2_t48_15B_UR50D']


class ESMWrapper:
    def __init__(self,
                 model_name='esm2_t33_650M_UR50D',
                 repr_layers='0,3,6,12,24,32,33',
                 include='logits,mean,per_tok,contacts',
                 extract_script='sh /home/hew/python/AggNet/framework/module/esm/esm2/extract.sh'):
        self.model_name = model_name
        self.repr_layers = repr_layers
        self.include = include
        self.extract_script = extract_script

    def get_command(self, input, output):
        return f'{self.extract_script} {self.model_name} {input} {output} {self.repr_layers} {self.include}'

    def extract(self, input, output):
        command = self.get_command(input, output)
        return os.system(command)
