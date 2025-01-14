import pytest

skorch = pytest.importorskip("skorch")
torch = pytest.importorskip("torch")

from distributed import Client
from distributed.protocol import deserialize, serialize


def test_serialize_deserialize_skorch_model():

    client = Client(processes=True, n_workers=1)

    class MyModule(torch.nn.Module):
        def __init__(self, num_units=10):
            super().__init__()
            self.dense0 = torch.nn.Linear(20, num_units)

        def forward(self, X, **kwargs):
            return self.dense0(X)

    net = skorch.NeuralNetClassifier(
        MyModule,
        max_epochs=10,
        iterator_train__shuffle=True,
    )

    def test_serialize_skorch(net):
        net = net.initialize()
        return deserialize(*serialize(net))

    # We test on a different worker to ensure that
    # errors skorch serialization faces on a different process
    # other than the client due to lack of __main__ context
    # are actually resolved
    # See isssue for context
    # https://github.com/dask/dask-ml/issues/549#issuecomment-669924762

    deserialized_net = list(client.run(test_serialize_skorch, net).values())[0]
    assert isinstance(deserialized_net.module_, MyModule)

    client.close()
