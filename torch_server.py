import torch
import mlflow
import yaml
import fedml
from fedml.runner import FedMLRunner
from fedml.data.MNIST.data_loader import download_mnist, load_partition_data_mnist


def load_data(args):
    download_mnist(args.data_cache_dir)
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)

    (
        client_num,
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ) = load_partition_data_mnist(
        args,
        args.batch_size,
        train_path=args.data_cache_dir + "/MNIST/train",
        test_path=args.data_cache_dir + "/MNIST/test",
    )

    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        class_num,
    ]
    return dataset, class_num


class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5001")
    mlflow.set_experiment('fedml_experiment')

    with mlflow.start_run(
            run_name='server',
            tags={"role": "server"}
    ) as run:
        mlflow.set_experiment_tag("version", "v1")

        with open('config/fedml_config.yaml') as f:
            file = yaml.full_load(f)
            mlflow.log_param(key='epochs', value=file['train_args']['epochs'])
            mlflow.log_param(key='federated_optimizer', value=file['train_args']['federated_optimizer'])
            mlflow.log_param(key='client_num_in_total', value=file['train_args']['client_num_in_total'])
            mlflow.log_param(key='client_num_per_round', value=file['train_args']['client_num_per_round'])
            mlflow.log_param(key='comm_round', value=file['train_args']['comm_round'])

        args = fedml.init()
        device = fedml.device.get_device(args)
        dataset, output_dim = load_data(args)

        model = LogisticRegression(28 * 28, output_dim)

        mlflow.pytorch.log_model(model, "model")
        mlflow.register_model(
            model_uri="runs:/{}/model".format(run.info.run_id),
            name="Model-Server",
        )

        fedml_runner = FedMLRunner(args, device, dataset, model)
        fedml_runner.run()

    mlflow.end_run()
