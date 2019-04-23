import json

import requests


class SmartulaAsk:
    def __init__(self, username, password, server_url):
        self._server_url = server_url if server_url.endswith('/') else server_url.join('/')
        self._token = self.__get_token(username, password, server_url)

    @staticmethod
    def __get_token(username, password, server_url):
        payload = {'username': username, 'password': password}
        r = requests.post(server_url + '/api/web/account/auth', json=payload)
        r.raise_for_status()
        return json.loads(r.text)['token']

    def get_token(self):
        return self._token
