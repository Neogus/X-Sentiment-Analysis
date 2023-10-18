import pandas as pd
import seaborn as sns # This library improves the graphics
sns.set_style(rc={'patch.edgecolor': '4E4E4E'}) # Activates a style
pd.options.display.max_colwidth = 300
from SA_Fun import *

auth, api, analyzer = initialize() #Authenticate Account
mode_dic = {'everyone': sentiment_by_influence(tweet_count, api), 'influencers': sentiment_by_community(time_limit, output_file, auth)}
dfb = pd.DataFrame(columns=['Datetime', 'Minutes', 'Filtered Tweets', 'Tweets per Min', 'P.Strength','P.Index','N.Strength','N.Index','C.Strength','S.Index','Price'])
dfc = pd.DataFrame(columns=['Datetime', 'P.Signal', 'S.Signal' ])
dfd = pd.DataFrame(columns=['Datetime', 'TS Signal', 'PS Signal', 'NS Signal','CS Signal', 'PI Signal','NI Signal', 'CI Signal','SR Signal','Price'])

for x in range(300):
    df = mode_dic[ta_mode](time_limit, output_file, auth)
    dfx_1, dfb, dfc, dfd = sentiment_analysis(df, filters, dfb, dfc, dfd, analyzer)
    backup(save_min, dfb, dfc, dfd)
    # clear_output(wait=True)
    plot(dfd)
    dfd = backtest(dfb, dfd)
    time.sleep(1)



