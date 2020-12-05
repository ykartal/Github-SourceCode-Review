from __future__ import unicode_literals
from service.GitHubPulls import GitHubPulls
import csv
import configparser

config = configparser.ConfigParser()
config.read('config.ini')
githubToken = config['DEFAULT']['githubToken']

with open('project_names_owners.csv') as inputCsv:
    with open('codeReviewDataset.csv', mode='w', newline='', encoding="utf-8") as csvFile:
        csvFileWriter = csv.writer(csvFile, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        csv_reader = csv.reader(inputCsv, delimiter=',')
        index = 0
        for row in csv_reader:
            index = index + 1
            print(index, ": ", row[1], "/", row[0]);
            jsonList = GitHubPulls.listPullCommentsRequests(GitHubPulls, row[1], row[0], githubToken)
            if len(jsonList)>0:
                    for comment in jsonList:
                        diff_hunk = comment['diff_hunk'].splitlines()
                        sourceCode = diff_hunk[len(diff_hunk)-1][1:].strip()
                        single_comment = comment['body'].strip().replace("\n", " ").replace("\r", " ");
                        #∟print([sourceCode.strip(), comment['body'].strip().replace("\n", " ").replace("\r", " ")])
                        csvFileWriter.writerow([sourceCode, single_comment, row[1]], row[0])
        
                
                