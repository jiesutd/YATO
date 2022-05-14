from transformers import BertConfig, BertTokenizer, BertModel, AlbertConfig, AlbertTokenizer, AlbertModel
import importlib


class CustomModel():
    def __init__(self, customCofig, customTokenizer, customModel):
        """

        :param customCofig:
        :param customTokenizer:
        :param customModel:
        """
        self.customCofig = customCofig
        self.customTokenizer = customTokenizer
        self.customModel = customModel
        self.configuration = None

    def get_bertmodel(self):
        if self.customCofig is None or self.customCofig.lower() == 'none':
            self.configuration = BertConfig()
        else:
            self.configuration = importlib.import_module(self.customCofig)

        model = BertModel(self.configuration)
        return model

    def bertmodel(self):
        if self.customCofig is None or self.customCofig.lower() == 'none':
            self.configuration = BertConfig()
        else:
            self.configuration = importlib.import_module(self.customCofig)

        model = BertModel(self.configuration)
        return model

    def berttokenizer(self, pretrain_weight):
        tokenizer = BertTokenizer.from_pretrained(pretrain_weight)
        return tokenizer

    def albertmodel(self):
        if self.customCofig is None or self.customCofig.lower() == 'none':
            self.configuration = AlbertConfig()
        else:
            self.configuration = importlib.import_module(self.customCofig)

        model = AlbertModel(self.configuration)
        return model

    def alberttokenizer(self, pretrain_weight):
        tokenizer = AlbertTokenizer.from_pretrained(pretrain_weight)
        return tokenizer
