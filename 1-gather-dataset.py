from __future__ import unicode_literals
from service.GitHubPulls import GitHubPulls
import csv
import configparser
import time

config = configparser.ConfigParser()
config.read('config.ini')
githubToken = config['DEFAULT']['githubToken']

time_start = time.time()
request_count = 0
with open('dataset/project_names_owners.csv') as inputCsv:
    with open('dataset/codeReviewDataset.csv', mode='w', newline='', encoding="utf-8") as csvFile:
        csvFileWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csvFileWriter.writerow(['diff_hunk','review','reviewer_id','reviewer_username','reviewer_type','reviewer_site_admin','owner','repo'])
        csv_reader = csv.reader(inputCsv, delimiter=',')
        index = 0
        total_count=0
        for row in csv_reader:
            index = index + 1
            if index < 0:  #update counter to continue, 0 to restart from scratch
                continue
            
            #blocking to script to not ban by Github. Github allows to max 5000 requests per hour
            time_now = time.time()
            time_diff = time_now - time_start
            if time_diff<=3600:
                if request_count >= 5000:
                    time.sleep(3600-time_diff)
                    request_count = 0
                    time_start = time.time()
            else:
                time_start = time.time()
                request_count = 0
            
            page = 1
            while True:
                commentList = GitHubPulls.listPullCommentsRequests(GitHubPulls, row[1], row[0], page, githubToken)
                total_count = len(commentList)+total_count
                print(index, ": ", row[1], "/", row[0], "/CommentCount: ", len(commentList), " / Page: ", page, "Total: ", total_count);
                if len(commentList)>0:
                    page = page + 1
                    for comment in commentList:
                        single_comment = comment['body'].strip().replace("\n", " ").replace("\r", " ");
                        diff_hunk = comment['diff_hunk'].splitlines()
                        sourceCode = diff_hunk[len(diff_hunk)-1][1:].strip()
                        i=2
                        while not sourceCode:
                            sourceCode = diff_hunk[len(diff_hunk)-i][1:].strip()
                            i = i+1
                        # sourceCode=comment['diff_hunk'].strip().replace("\n", " ").replace("\r", " ");
                        reviewer=comment['user']
                        if reviewer is not None:
                            reviewer_id=reviewer['id']
                            reviewer_username= reviewer['login']
                            reviewer_type= reviewer['type']
                            reviewer_site_admin= reviewer['site_admin']
                            csvFileWriter.writerow([sourceCode, single_comment, reviewer_id, reviewer_username, reviewer_type, reviewer_site_admin, row[1], row[0]])
                else:
                    break
                
                