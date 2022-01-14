from component.client import Client


class BAClient(Client):
    def __init__(self, client_id, local_dataloader, local_data_size, malicious=False):
        super(BAClient, self).__init__(client_id, local_dataloader, local_data_size)
        self.malicious = malicious
        self.mal_data_loader = None
