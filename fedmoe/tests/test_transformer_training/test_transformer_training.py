import torch

from experiments.utils import load_data
from fedmoe.client_manager import PreTrainingClientManager
from fedmoe.game.transformer_game import TransformerGame
from fedmoe.server import Server

torch.set_default_dtype(torch.float64)

TOTAL_ROUNDS = 10
data_object = load_data("periodic_signal", TOTAL_ROUNDS + 1)
Z_DIM = 4
T = 3
ALPHA = 0.1
GAMMA = 0.1
Y_DIM = 1


def test_inference_periodic_data_transformer() -> None:
    """
    The goal of this test to train the transformer and make it over-fit on the data to see actually if
    it is able to have good predictions with a lot of training. The goal of the algorithm is not to use the best
    transformer, but to see if the structure and training is suitable for this dataset at least. We can tune
    the transformer structure and change the training for other datasets if needed.
    # No assertion is used, we manually validate the predictions by printing them.
    """
    data_loader = data_object.get_dataloader(num_samples=20, batch_size=4)
    # I checked and input and target make sense (in inference data and data loader)

    client_manager = PreTrainingClientManager(
        num_clients=1,
        data_sequence=data_object.input_matrix,
        sync_freq=T,
        z_dim=Z_DIM,
        alpha=ALPHA,
        gamma=GAMMA,
        pre_training_dataloader=data_loader,
        pre_training_epochs=800,
        pre_training_learning_rate=0.01,
        target_sequence=data_object.target_matrix,
    )

    game = TransformerGame(
        client_manager.clients,
        sync_freq=T,
        z_dim=Z_DIM,
    )

    _ = Server(
        total_game_steps=T,
        client_manager=client_manager,
        game=game,
        metrics=[],
        kappa=1.0,
        eta=1.0,
    )

    # test the transformer
    client_manager.clients[0].encoder.eval()
    input_0 = data_object.input_matrix[0].reshape(1, -1, 1)
    expected_target_0 = data_object.target_matrix[1]
    pred_0 = client_manager.clients[0].encoder(input_0, pre_training=True)
    print("input_0", input_0)
    print("expected_target_0", expected_target_0)
    print("pred_0", pred_0)

    input_1 = data_object.input_matrix[1].reshape(1, -1, 1)
    expected_target_1 = data_object.target_matrix[2]
    pred_1 = client_manager.clients[0].encoder(input_1, pre_training=True)
    print("\ninput_1", input_1)
    print("expected_target_1", expected_target_1)
    # Note that this prediction will not be that close because we're putting input_1 in the first position
    # and the transformer is used to seeing input_0 there
    print("pred_1", pred_1)

    input_2 = data_object.input_matrix[2].reshape(1, -1, 1)
    pred_2 = client_manager.clients[0].encoder(input_2, pre_training=True)
    print("\ninput_2", input_2)
    print("expected_target_2", data_object.target_matrix[3])
    # Note that this prediction will not be that close because we're putting input_2 in the first position
    # and the transformer is used to seeing input_0 there
    print("pred_2", pred_2)

    pred_0_1_input2 = torch.cat((input_0, input_1, input_2), dim=1)
    pred_2 = client_manager.clients[0].encoder(pred_0_1_input2, pre_training=True)
    print("\nInput shape (batch_size, sequence length, embedding dim)", pred_0_1_input2.shape)
    print("expected_target_2", data_object.target_matrix[3])
    print("full pred 2", pred_2)
    # Because this prediction is part of a sequence that the model has seen many times. This value should be close
    print("pred_2", pred_2[:, -1, :])

    input_3 = data_object.input_matrix[3].reshape(1, -1, 1)
    pred_0_1_input3 = torch.cat((input_0, input_1, input_2, input_3), dim=1)
    pred_3 = client_manager.clients[0].encoder(pred_0_1_input3, pre_training=True)
    print("\ninput_3", pred_0_1_input3)
    print("target 3", data_object.target_matrix[4])
    print("Sequence Prediction", pred_3)
    # Because this prediction is part of a sequence that the model has seen many times. This value should be close
    print("pred_3", pred_3[:, -1, :])

    input_4 = data_object.input_matrix[4].reshape(1, -1, 1)
    pred_4_alone = client_manager.clients[0].encoder(input_4, pre_training=True)
    print("\ntarget  4", data_object.target_matrix[5])
    # Note that this prediction might not be that close because we're putting input_4 in the first position
    # and the transformer is used to seeing input_0 there
    print("solo pred 4", pred_4_alone)
    full_seq_4 = torch.cat((pred_0_1_input3, input_4), dim=1)
    pred_4_seq = client_manager.clients[0].encoder(full_seq_4, pre_training=True)
    # Because this prediction is part of a sequence that the model has seen many times. This value should be close
    print("full pred 4,", pred_4_seq)

    # assert False
