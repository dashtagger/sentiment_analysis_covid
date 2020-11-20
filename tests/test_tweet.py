import os

import pytest

from src.modules.tweet_module import Twitter_api

class TestTweet:
    def test_tokens(self):
        with pytest.raises(ValueError) as e:
            api = Twitter_api()
        err_msg = 'Provide the correct tokens and keys'
        assert e.match(err_msg), 'All credentials are given and accepted'
