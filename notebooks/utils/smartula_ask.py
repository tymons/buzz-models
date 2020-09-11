import json
import requests
import csv

class SmartulaAsk:
    def __init__(self, username, password, server_url):
        self._server_url = server_url if server_url.endswith('/') else server_url.join('/')
        self._token = self.__get_token(username, password, server_url)

    @staticmethod
    def __get_token(username, password, server_url):
        payload = {'username': username, 'password': password}
        r = requests.post(server_url + 'api/web/account/auth', json=payload)
        r.raise_for_status()
        return json.loads(r.text)['token']

    def get_token(self):
        return self._token
    
    def get_temperatures(self, hive_sn, from_timestamp, to_timestamp):
        """Get temperatures in range.
            :param hive_sn: hive sn
            :param from_timestmap: from_timestmap
            :type to_timestamp: to_timestamp
            :return: list of tuples [(timestamp,value)]
        """
        url = f"{self._server_url}api/web/measurement/temperature/all/{hive_sn}/"
        params = {"from": from_timestamp, "to": to_timestamp}
        headers = {"token" : self._token}
        response = requests.get(url, headers = headers, params = params)
        response = json.loads(response.text)
        return [(measurement['timestampMax'].replace(":", "-"), measurement['avgValue']) for measurement in response]
        
    def get_humidities(self, hive_sn, from_timestamp, to_timestamp):
        """Get humidity in range.
            :param hive_sn: hive sn
            :param from_timestmap: from_timestmap
            :type to_timestamp: to_timestamp
            :return: list of tuples [(timestamp,value)]
        """
        url = f"{self._server_url}api/web/measurement/humidity/all/{hive_sn}/"
        params = {"from": from_timestamp, "to": to_timestamp}
        headers = {"token" : self._token}
        response = requests.get(url, headers = headers, params = params)
        response = json.loads(response.text)
        return [(measurement['timestampMax'].replace(":", "-"), measurement['avgValue']) for measurement in response]

    def get_sound(self, hive_sn, sound_id):
        url = self._server_url + 'api/web/measurement/sound/new/' + str(hive_sn)
        params = {'soundId': sound_id}
        headers = {'token': self._token}
        response = requests.get(url, headers=headers, params=params)
        response = json.loads(response.text)
        return response['samples'], response['timestamp']

    def get_sound_ids(self, hive_sn, from_timestamp, to_timestamp):
        url = f"{self._server_url}api/web/measurement/sound/{hive_sn}/getAvailableSounds/"
        params = {"from": from_timestamp, "to": to_timestamp}
        headers = {"token" : self._token}
        response = requests.get(url, headers = headers, params = params)
        response = json.loads(response.text)
        return list(response.values())
                