from framework.data import ProteinDataset


class DatasetManager:
    def __init__(self):
        super(DatasetManager, self).__init__()

        self.protein_dataset = ['RationalDesignProtien', 'Nb_b201', 'Nb_H11_D4', 'Nb_H11_H4', 'Amy17', 'Amy37', 'Amyl37', 'ALBase', 'ALBase678',
                                'FoldAgg77', 'IAPP8', 'Sol2151', 'A3DHumanDB']
        self.scFv_datasets = ['A3D_OAS', 'HzATNP', 'Adalimumab', 'Golimumab', 'CR3022', 'CST137', 'Shehata400', 'Roche17',
                              'TPBLA_WFL', 'TPBLA_JTO', 'Lai_mAbs21', 'Lai_mAbs27']
        self.peptide_dataset = ['Hexapeptide', 'CPAD2.0', 'Adnectin31', 'Cordax96']
        self.mutation_dataset = ['SoluProtMutDB', 'FireProtDB']
        self.all_datasets = self.protein_dataset + self.scFv_datasets + self.peptide_dataset + self.mutation_dataset
        self.cache = {}

    def load_dataset(self, dataset):
        if dataset in self.all_datasets:
            if dataset in self.cache:
                return self.cache[dataset]
            else:
                data_class = 'Antibody' if dataset in self.scFv_datasets else 'Protein'
                print(f'Load Dataset: [{dataset}], Class: [{data_class}]')
                dataset = ProteinDataset.ProteinDataset(dataset)
                dataset.load(data_class=data_class)
                self.cache[dataset.name] = dataset
        else:
            raise NotImplementedError(f'No such dataset: {dataset}')
        return dataset
