from fedmoe.client_manager import ClientManager
from fedmoe.clients.client import ClientType
from fedmoe.game import EchoStateGame, RfnGame, TransformerGame
from fedmoe.server import Server


def run_rfn_game(sync_freq: int, d_z: int) -> None:
    client_manager = ClientManager(
        client_type=ClientType.RFN,
        num_clients=5,
        sync_freq=sync_freq,
        d_z=d_z,
        alpha=0.1,
        gamma=0.1,
    )
    game = RfnGame(
        client_manager.clients,
        sync_freq=sync_freq,
        d_z=d_z,
    )
    server = Server(sync_freq=sync_freq, client_manager=client_manager, game=game)
    server.fit(70)


def run_transformer_game(sync_freq: int, d_z: int) -> None:
    client_manager = ClientManager(
        client_type=ClientType.TRANSFORMER,
        num_clients=3,
        sync_freq=sync_freq,
        d_z=d_z,
        alpha=0.1,
        gamma=0.1,
    )
    game = TransformerGame(
        client_manager.clients,
        sync_freq=sync_freq,
        d_z=d_z,
    )
    server = Server(sync_freq=sync_freq, client_manager=client_manager, game=game)
    server.fit(70)


def run_esn_game(sync_freq: int, d_z: int) -> None:
    client_manager = ClientManager(
        client_type=ClientType.ESN,
        num_clients=3,
        sync_freq=sync_freq,
        d_z=d_z,
        alpha=0.1,
        gamma=0.1,
    )
    game = EchoStateGame(
        client_manager.clients,
        sync_freq=sync_freq,
        d_z=d_z,
        N_samples=200,
    )
    server = Server(sync_freq=sync_freq, client_manager=client_manager, game=game)
    server.fit(70)


if __name__ == "__main__":
    sync_freq: int = 7
    d_z: int = 8

    # run_rfn_game(sync_freq, d_z)

    run_transformer_game(sync_freq, d_z)

    # run_esn_game(sync_freq, d_z)
