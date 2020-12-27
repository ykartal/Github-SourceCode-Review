
import requests
class GitHubPulls:
    url = 'https://api.github.com'
    def _repr(self):
        return "<GitHubPulls [{0}]>".format(self.path)

    def __init__(self):
        self.url = self.url
        return self
        
    def listPullCommentsRequests(self, owner, repo, page, access_token):
        pullGet = self.url + "/repos/" + owner + "/" + repo + "/pulls/comments?page="+str(page)
        authorization_header = {"Authorization": "token %s" % access_token}
        req = requests.get(pullGet, headers=authorization_header)
        if req.status_code == 200:
            return req.json();
        else:
            link = req.links
            return link

