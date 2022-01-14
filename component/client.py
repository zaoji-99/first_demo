class Client:
    def __init__(self, client_id, local_dataloader, local_data_size):
        self.client_id = f'client_{client_id}'
        self.local_dataloader = local_dataloader
        self.local_data_size = local_data_size


