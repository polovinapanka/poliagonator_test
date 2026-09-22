import socket
import json


class UnrealClient:
    UE_IP = "127.0.0.1"
    UE_RECEIVE_PORT = 3002

    PYTHON_IP = "127.0.0.1"
    PYTHON_RECEIVE_PORT = 3001

    def __init__(self):
        self.sock = socket.socket(
            socket.AF_INET,
            socket.SOCK_DGRAM
        )
        self.sock.bind(
            (self.PYTHON_IP, self.PYTHON_RECEIVE_PORT)
        )
        self.sock.settimeout(5.0)

    def step(self, action: int):
        request = {
            "action": int(action)
        }

        data = json.dumps(request).encode("utf-8")

        self.sock.sendto(
            data,
            (self.UE_IP, self.UE_RECEIVE_PORT)
        )

        response_data, _ = self.sock.recvfrom(65535)

        response = json.loads(
            response_data.decode("utf-8")
        )

        return response

    def close(self):
        self.sock.close()