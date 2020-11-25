
import requests
class GitHubPulls:
    url = 'https://api.github.com'
    def _repr(self):
        return "<GitHubPulls [{0}]>".format(self.path)

    def __init__(self):
        self.url = self.url
        return self
        
    def listPullCommentsRequests(self, owner, repo, access_token):
        pullGet = self.url + "/repos/" + owner + "/" + repo + "/pulls/comments"
        authorization_header = {"Authorization": "token %s" % access_token}
        if requests.get(pullGet).status_code == 200:
            return requests.get(pullGet, headers=authorization_header).json();
        else:
            link = requests.get(pullGet).links
            return link

