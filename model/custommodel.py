from transformers import BertConfig, BertTokenizer, BertModel, AlbertConfig, AlbertTokenizer, AlbertModel
import importlib


class CustomModel():
    def __init__(self, customConfig, customTokenizer, customModel):

        self.customConfig = customConfig
        self.customTokenizer = customTokenizer
        self.customModel = customModel
        self.configuration = None

    def get_bertmodel(self):
        if self.customConfig is None or self.customConfig.lower() == 'none':
            self.configuration = BertConfig()
        else:
            self.configuration = importlib.import_module(self.customConfig)

        model = BertModel(self.configuration)
        return model

    def bertmodel(self):
        if self.customConfig is None or self.customConfig.lower() == 'none':
            self.configuration = BertConfig()
        else:
            self.configuration = importlib.import_module(self.customConfig)

        model = BertModel(self.configuration)
        return model

    def berttokenizer(self, pretrain_weight):
        tokenizer = BertTokenizer.from_pretrained(pretrain_weight)
        return tokenizer

    def albertmodel(self):
        if self.customConfig is None or self.customConfig.lower() == 'none':
            self.configuration = AlbertConfig()
        else:
            self.configuration = importlib.import_module(self.customConfig)

        model = AlbertModel(self.configuration)
        return model

    def alberttokenizer(self, pretrain_weight):
        tokenizer = AlbertTokenizer.from_pretrained(pretrain_weight)
        return tokenizer
