from json import encoder
import snscrape.modules.twitter as twitterScraper
import json
import unicodedata as ud

safe = {
            
    }

names = [""]#Profile usernames to scrape.
for name in names:
    
    scrapper = twitterScraper.TwitterUserScraper(name,False)
    tweets = []
    d = {ord('\N{COMBINING ACUTE ACCENT}'):None}
                
    for i,tweet in enumerate(scrapper.get_items()):
        #if i>1000:
        #    break
        text= ud.normalize('NFD',tweet.content).translate(d)
        #print(tweet.inReplyToTweetId)
        if(tweet.inReplyToTweetId is  None):
            if(True):
                for word in safe:
                    if(ud.normalize('NFD',word).translate(d) in text):
                        #print(i,word)
                        tweets.append({
                        "followers":tweet.user.followersCount,
                        "description":tweet.user.description,
                        "verified":tweet.user.verified,
                        "created":str(tweet.user.created),
                        "friends":tweet.user.friendsCount,
                        "favorites":tweet.user.favouritesCount,
                        "content": tweet.content,
                        "likes": tweet.likeCount,
                        "replies": tweet.replyCount,
                        "retweets": tweet.retweetCount,
                        "quotes": tweet.quoteCount,
                        })
                        break
    print(len(tweets))

    #pass data to json file
    f = open(name+".json","w",encoding="utf-8")
    j = json.dumps(tweets,ensure_ascii=False)
    f.write(j)
    f.close()
