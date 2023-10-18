from datetime import datetime, timedelta
import ccxt
#from keras.models import Sequential, load_model
#from keras.layers import LSTM, Dense, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import tweepy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.request, json
from datetime import datetime, timedelta
import ccxt
#from keras.models import Sequential, load_model
#from keras.layers import LSTM, Dense, Activation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tweepy
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import urllib.request, json
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import sys
import csv
import seaborn as sns # This library improves the graphics
import statistics
sns.set_style(rc={'patch.edgecolor': '4E4E4E'}) # Activates a style
pd.options.display.max_colwidth = 300
from SA_Keys import *
from SA_Config import *
import pandas as pd
import seaborn as sns # This library improves the graphics
sns.set_style(rc={'patch.edgecolor': '4E4E4E'}) # Activates a style
pd.options.display.max_colwidth = 300


def price_matrix_creator(data, seq_len=30):
    '''
    It converts the series into a nested list where every item of the list contains historic prices of 30 days
    '''
    price_matrix = []
    for index in range(len(data) - seq_len + 1):
        price_matrix.append(data[index:index + seq_len])
    return price_matrix


def normalize_windows(window_data):
    '''
    It normalizes each value to reflect the percentage changes from starting point
    '''
    normalised_data = []
    for window in window_data:
        normalised_window = [((float(p) / float(window[0])) - 1) for p in window]
        normalised_data.append(normalised_window)
    return normalised_data


def train_test_split_(price_matrix, train_size=0.9, shuffle=False, return_row=True):
    '''
    It makes a custom train test split where the last part is kept as the training set.
    '''
    price_matrix = np.array(price_matrix)
    # print(price_matrix.shape)
    row = int(round(train_size * len(price_matrix)))
    train = price_matrix[:row, :]
    if shuffle == True:
        np.random.shuffle(train)
    X_train, y_train = train[:row, :-1], train[:row, -1]
    X_test, y_test = price_matrix[row:, :-1], price_matrix[row:, -1]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    if return_row:
        return row, X_train, y_train, X_test, y_test
    else:
        X_train, y_train, X_test, y_test


def deserializer(preds, data=[], train_size=0.9, train_phase=False):
    '''
    Arguments:
    preds : Predictions to be converted back to their original values
    data : It takes the data into account because the normalization was made based on the full historic data
    train_size : Only applicable when used in train_phase
    train_phase : When a train-test split is made, this should be set to True so that a cut point (row) is calculated based on the train_size argument, otherwise cut point is set to 0

    Returns:
    A list of deserialized prediction values, original true values, and date values for plotting
    '''
    price_matrix = np.array(price_matrix_creator(data))
    if train_phase:
        row = int(round(train_size * len(price_matrix)))
    else:
        row = 0
    date = data.index[row + 29:]
    date = np.reshape(date, (date.shape[0]))
    X_test = price_matrix[row:, :-1]
    y_test = price_matrix[row:, -1]
    preds_original = []
    preds = np.reshape(preds, (preds.shape[0]))
    for index in range(0, len(preds)):
        pred = (preds[index] + 1) * X_test[index][0]
        preds_original.append(pred)
    preds_original = np.array(preds_original)
    if train_phase:
        return [date, y_test, preds_original]
    else:
        import datetime
        return [date + datetime.timedelta(days=1), y_test]


def retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    num_retries = 0
    try:
        num_retries += 1
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)
        # print('Fetched', len(ohlcv), symbol, 'candles from', exchange.iso8601 (ohlcv[0][0]), 'to', exchange.iso8601 (ohlcv[-1][0]))
        return ohlcv
    except Exception:
        if num_retries > max_retries:
            raise  # Exception('Failed to fetch', timeframe, symbol, 'OHLCV in', max_retries, 'attempts')


def scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit):
    earliest_timestamp = exchange.milliseconds()
    timeframe_duration_in_seconds = exchange.parse_timeframe(timeframe)
    timeframe_duration_in_ms = timeframe_duration_in_seconds * 1000
    timedelta = limit * timeframe_duration_in_ms
    all_ohlcv = []
    while True:
        fetch_since = earliest_timestamp - timedelta
        ohlcv = retry_fetch_ohlcv(exchange, max_retries, symbol, timeframe, fetch_since, limit)
        # if we have reached the beginning of history
        if ohlcv[0][0] >= earliest_timestamp:
            break
        earliest_timestamp = ohlcv[0][0]
        all_ohlcv = ohlcv + all_ohlcv
        # print(len(all_ohlcv), 'candles in total from', exchange.iso8601(all_ohlcv[0][0]), 'to', exchange.iso8601(all_ohlcv[-1][0]))
        # if we have reached the checkpoint
        if fetch_since < since:
            break
    return exchange.filter_by_since_limit(all_ohlcv, since, None, key=0)


def scrape_candles_to_csv(filename, exchange_id, max_retries, symbol, timeframe, since, limit):
    # instantiate the exchange by id
    exchange = getattr(ccxt, exchange_id)({
        'enableRateLimit': True,
    })
    # convert since from string to milliseconds integer if needed
    if isinstance(since, str):
        since = exchange.parse8601(since)
    # preload all markets from the exchange
    exchange.load_markets()
    # fetch all candles
    ohlcv = scrape_ohlcv(exchange, max_retries, symbol, timeframe, since, limit)
    # Creates Dataframe and save it to csv file
    pd.DataFrame(ohlcv).to_csv(f'{filename}')
    # print('Saved', len(ohlcv), 'candles from', exchange.iso8601(ohlcv[0][0]), 'to', exchange.iso8601(ohlcv[-1][0]), 'to', filename)


def import_csv(filename='_Crypto12-Index|binance|1d|2020-01-01T00:00:00Z--2020-07-25T01:22:39|UTC',
               csv_folder='/home/a/Documents/Data Science Projects/Trading Algorithms/Raw CSVs/'):
    df = pd.read_csv(csv_folder + filename, index_col='Datetime', parse_dates=True)  # .tz_convert(tz=None)
    return df


def fetch_for_montecarlo(exchange='binance',
                         cryptos=['BTC/USDT', 'ETH/USDT', 'XRP/USDT', 'BCH/USDT', 'ADA/USDT', 'LTC/USDT',
                                  'LINK/USDT', 'EOS/USDT', 'XLM/USDT', 'XTZ/USDT', 'XMR/USDT', 'VET/USDT'],
                         sample_freq='1d', since_days=28, page_limit=1000):
    datetime_now = datetime.now().strftime('%Y-%m-%d')
    since = (datetime.today() - timedelta(days=since_days)).strftime('%Y-%m-%dT%H:%M:%S')
    print('Begin download...')
    for market_symbol in cryptos:
        scrape_candles_to_csv(filename='test', exchange_id=exchange, max_retries=3, symbol=market_symbol,
                              timeframe=sample_freq, since=since, limit=page_limit)
        time.sleep(2)
        df = pd.read_csv('test')

        if market_symbol == cryptos[0]:

            df.drop(df.columns[[0, 2, 3, 4, 6]], axis=1, inplace=True)
            df['0'] = pd.to_datetime(df['0'], unit='ms')
            df.rename(columns={'0': 'Datetime', '4': f'{market_symbol} Close'}, inplace=True)
            df = df.set_index('Datetime')
            dfx = df.copy()

        else:

            df.drop(df.columns[[0, 2, 3, 4, 6]], axis=1, inplace=True)
            df['0'] = pd.to_datetime(df['0'], unit='ms')
            df.rename(columns={'0': 'Datetime', '4': f'{market_symbol} Close'}, inplace=True)
            df = df.set_index('Datetime')
            dfx = pd.merge(dfx, df, on=['Datetime'])

    dfx = dfx.loc[:, ~dfx.columns.duplicated()]
    dfx = dfx[~dfx.index.duplicated(keep='first')]
    market_symbol = market_symbol.replace('/', '-')

    csv_name = f'_Montecarlo|{exchange}|{sample_freq}|{datetime_now} UTC'
    print('Finished')
    return datetime_now, dfx.to_csv('/home/a/Documents/Data Science Projects/Trading Algorithms/Raw CSVs/' + csv_name)


def rnn_day(df, model_save='coin_predictor.h3', symbol='BTC', base='BUSD'):
    ser = df.resample('D').mean().fillna(method='ffill')[f'{symbol}/{base} Close']

    price_matrix = price_matrix_creator(ser)
    X_test = normalize_windows(price_matrix)
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    model = load_model(f'{model_save}')
    preds = model.predict(X_test, batch_size=2)
    final_pred = deserializer(preds, ser, train_size=0.9, train_phase=False)
    print(f'Price for tomorrow = {final_pred[1][0]}')


def sharp_calc(df=[], expectancy=24):
    for col in df.columns:

        ret = df[col].pct_change(1).drop(df.index[0]).mean()
        vol = df[col].pct_change(1).drop(df.index[0]).std()

        if vol != 0:
            sharpe = (expectancy ** 0.5) * (ret / vol)
        else:
            sharpe = None

        exp_price = (1 + ret * expectancy) * df[col][-1]

        print(f'{col} SR = {sharpe:.2f}')
        print(f'{col} Exp.Price = {exp_price:.4f}')


# ETL for modeling:

def rnn_model(df=[], batch_size=2, epochs=15, seq_len=30, loss='mean_squared_error', optimizer='rmsprop',
              activation='linear', input_shape=(None, 1), output_dim=30, output_name='coin_predictor'):
    ser = df.resample('D').mean().fillna(method='ffill')['BTC/USDT Close']

    price_matrix = price_matrix_creator(ser)  # Creating a matrix using the dataframe
    price_matrix = normalize_windows(price_matrix)  # Normalizing its values to fit to RNN
    row, X_train, y_train, X_test, y_test = train_test_split_(
        price_matrix)  # Applying train-test splitting, also returning the splitting-point

    # Model Configuration:

    model = Sequential()
    model.add(LSTM(units=output_dim, return_sequences=True, input_shape=input_shape))
    model.add(Dense(units=32, activation=activation))
    model.add(LSTM(units=output_dim, return_sequences=False))
    model.add(Dense(units=1, activation=activation))
    model.compile(optimizer=optimizer, loss=loss)

    # Create Model:
    print('Fitting Model...')
    start_time = time.time()
    model.fit(x=X_train,
              y=y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.05)
    end_time = time.time()
    processing_time = end_time - start_time
    print('Model completed')
    # Save Model:

    model.save(f'{output_name}')
    print(f'Model saved as:{output_name}')


'''
class MyStreamListener(tweepy.StreamListener):
    def __init__(self, time_limit=60):
        self.start_time = time.time()
        self.limit = time_limit
        self.saveFile = open('abcd.json', 'a')
        super(MyStreamListener, self).__init__()

    def on_data(self, data):
        if (time.time() - self.start_time) < self.limit:
            self.saveFile.write(data)
            self.saveFile.write('\n')
            return True
        else:
            self.saveFile.close()
            return False
            '''
counter = 1


class MyStreamListener(tweepy.StreamingClient):
    """
    Twitter listener, collects streaming tweets and output to a file
    """

    def __init__(self, time_limit, output_file):
        super(MyStreamListener, self).__init__(bearer_token=access_key)
        self.num_tweets = 0
        self.start_time = time.time()
        self.limit = time_limit
        self.file = open(output_file, "w")

    def on_status(self, status):
        global counter
        try:
            text = status.extended_tweet["full_text"]
        except AttributeError:
            text = status.text
        # Writing status data
        with open(output_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([status.created_at, status.lang, status.author.screen_name, text])
        self.num_tweets += 1
        current_time = time.time() - self.start_time
        # checkpoint = counter * 30

        # Stops streaming when it reaches the limit
        if self.num_tweets <= tweet_count and tweet_count != 0:
            # if self.num_tweets % 100 == 0: # just to see some progress...
            # print(f'Numer of tweets captured so far: {self.num_tweets}')
            return True

        elif current_time <= time_limit and time_limit != 0:

            # if current_time > checkpoint: # just to see some progress...
            # print(f'Time elapsed: {math.floor(current_time)} seconds...')
            # counter += 1
            return True
        else:
            return False
        self.file.close()

    def on_error(self, status):
        print(status)

    def on_exception(self, exception):
        print(exception)
        return

    def on_timeout(self):
        print(sys.stderr, 'Timeout...')
        return True  # Don't kill the stream
        print("Stream restarted")


def remove_pattern(input_txt, pattern):
    r = re.findall(pattern, input_txt)
    for i in r:
        input_txt = re.sub(i, '', input_txt)
    return input_txt


def clean_tweets(tweets):
    # remove twitter Return handles (RT @xxx:)
    tweets = np.vectorize(remove_pattern)(tweets, "RT @[\w]*:")

    # remove twitter handles (@xxx)
    tweets = np.vectorize(remove_pattern)(tweets, "@[\w]*")

    # remove URL links (httpxxx)
    tweets = np.vectorize(remove_pattern)(tweets, "https?://[A-Za-z0-9./]*")

    # remove special characters, numbers, punctuations (except for #)
    tweets = np.core.defchararray.replace(tweets, "[^a-zA-Z]", " ")

    return tweets


def initialize():
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_key, access_secret)
    api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
    analyzer = SentimentIntensityAnalyzer()
    new_words = {'bear': -2, 'bull': 2, 'bullish': 2, 'bulling': 2, 'bearish': -2, 'bearing': -2, }

    analyzer.lexicon.update(new_words)

    return auth, api, analyzer

''' 
                           --- Sentiment by Community --- 
This function will create a tweet dataset from a stream filtered by the keywords,
   list, it reflects the twitter community sentiment. '''

def sentiment_by_community(auth, time_limit, output_file):
    global influence

    influence = False

    # Initialize Stream listener
    l = MyStreamListener(time_limit, output_file)

    # Create you Stream object with authentication
    stream = tweepy.StreamingClient(auth, l)

    # Filter Twitter Streams to capture data by the keywords:
    stream.filter(track=keywords)

    print('\n Finished Streaming \n')

    df = pd.read_csv(output_file, names=['created_at', 'lang', 'source', 'text'])
    df['text'] = clean_tweets(df['text'])

    return df

''' 
                           --- Sentiment by Influence --- 
This function will create a tweet dataset from cryptocurrency influencers accounts 
specified in the variable tweet_count, this represents the experts opinion. '''

def sentiment_by_influence(tweet_count, api):
    global influence

    influence = True

    # Array to hold sentiment

    sentiments = []

    # Iterate through all the users

    for search in influencers:

        # Bring out an count of tweets
        users_tweets = api.user_timeline(screen_name=search, count=tweet_count, tweet_mode='extended')


        # Loop through the tweets

        for tweet in users_tweets:
            text = tweet["full_text"]

            # Add each value to the appropriate array

            sentiments.append({"User": search,
                               "text": text,
                               "created_at": tweet["created_at"],
                               "lang": tweet["lang"],
                               "source": tweet["source"]
                               })

    # Convert array to dataframe

    df = pd.DataFrame.from_dict(sentiments)
    df['text'] = clean_tweets(df['text'])

    return df


def print_dfx(dfx):
    print(f"Positive Sum = {round(dfx['Positive'].sum(), 2)}")
    print(f"Positive Average = {round(dfx['Positive'].mean(), 2)}")
    print(f"Negative Sum = {round(dfx['Negative'].sum(), 2)}")
    print(f"Negative Average = {round(dfx['Negative'].mean(), 2)}")
    print(f"Accumulated Sentiment = {round(dfx['Compound'].sum(), 2)}")
    print(f"Sentiment Index = {round(dfx['Compound'].mean(), 3)} \n")

    return round(dfx['Compound'].mean(), 3), len(dfx)


def mean_pos(L):
    # Get all positive numbers into another list
    pos_only = [x for x in L if x > 0]
    if pos_only:
        return sum(pos_only) / len(pos_only)
    raise ValueError('No postive numbers in input')


def mean_neg(L):
    # Get all positive numbers into another list
    pos_only = [x for x in L if x > 0]
    if pos_only:
        return sum(pos_only) / len(pos_only)
    raise ValueError('No postive numbers in input')


def sentiment_analysis(df, filters, dfb, dfc, dfd, analyzer):
    # Declare variables for scores

    scores = []
    compound_list = []
    positive_list = []
    negative_list = []
    neutral_list = []

    # print(analyser.polarity_scores(sentiments_pd['text'][i]))

    for i in range(df['text'].shape[0]):
        compound = analyzer.polarity_scores(df['text'][i])["compound"]
        pos = analyzer.polarity_scores(df['text'][i])["pos"]
        neu = analyzer.polarity_scores(df['text'][i])["neu"]
        neg = analyzer.polarity_scores(df['text'][i])["neg"]

        scores.append({"Compound": compound,
                       "Positive": pos,
                       "Negative": neg,
                       "Neutral": neu
                       })

    sentiments_score = pd.DataFrame.from_dict(scores)
    df = df.join(sentiments_score)
    df = df[(df['text'] != '') & ~ (df['text'].str.contains('|'.join(filters), case=False))]
    df = df.drop_duplicates(subset="text", keep=False)
    df = df[df['lang'] == language]

    if influence is True:

        formato = '%a %b %d %H:%M:%S +%f %Y'
        df['Datetime'] = pd.to_datetime(df['created_at'], format=formato)
        datetime = pd.to_datetime(df['created_at']).sort_values().max().tz_localize(None)
        print(f'Dataset date = {datetime} \n')
        df = df.set_index(pd.DatetimeIndex(df['Datetime'])).drop(columns=['Datetime', 'created_at', 'lang'])
        dfx_1 = df[(df['text'].str.contains('|'.join(keywords), case=False)) & (df['Compound'] != 0) & (
                    df.index >= (datetime - dt.timedelta(minutes=since_min))) & (
                               df.index <= (datetime - dt.timedelta(minutes=int(since_min / 2))))].sort_index(
            ascending=False)
        print("First half Sentiment Scores: \n")
        first_sent, len_1 = print_dfx(dfx_1)
        dfx_1.index = dfx_1.index - pd.DateOffset(hours=3)
        dfx_2 = df[(df['text'].str.contains('|'.join(keywords), case=False)) & (df['Compound'] != 0) & (
                    df.index >= (datetime - dt.timedelta(minutes=int(since_min / 2))))].sort_index(ascending=False)
        print("Second half Sentiment Scores: \n")
        second_sent, len_2 = print_dfx(dfx_2)
        dfx_2.index = dfx_2.index - pd.DateOffset(hours=3)

        print(f'Sentiment % variation = {round(second_sent / first_sent - 1, 2) * 100}% \n')
        print(f'Tweets Count for First Period = {len(dfx_1)} \n')
        print(f'Tweets Count for Second Period = {len(dfx_2)} \n')
        print(f'Tweets Count % Variation = {round(len_2 / len_1 - 1, 2) * 100}% \n')

        return dfx_1, dfx_2

    else:

        formato = '%Y-%m-%d %H:%M:%S'
        df['Datetime'] = pd.to_datetime(df['created_at'], format=formato)
        datetime = pd.to_datetime(df['created_at']).sort_values().max().tz_localize(None)
        df = df.set_index(pd.DatetimeIndex(df['Datetime'])).drop(columns=['Datetime', 'created_at', 'lang'])
        dfx_1 = df[(df['text'].str.contains('|'.join(keywords), case=False)) & (df['Compound'] != 0) & (
                    df.index >= (datetime - dt.timedelta(minutes=since_min)))].sort_index(ascending=False)

        print_dfx(dfx_1)

        time_lapse = dfx_1.index.max() - dfx_1.index.min()
        t_count = len(dfx_1)
        print("Sentiment Scores: \n")
        print(f'Tweets Count = {t_count} \n')
        print(f'Dataset time window: {time_lapse} \n')
        dfx_1.index = dfx_1.index - pd.DateOffset(hours=3)

        t_min = time_lapse.total_seconds() / 60
        p_min = round(dfx_1['Positive'].sum() / t_min, 3)
        p_avg = round(dfx_1['Positive'].mean(), 3)
        n_min = round(dfx_1['Negative'].sum() / t_min, 3)
        n_avg = round(dfx_1['Negative'].mean(), 3)
        c_min = round(dfx_1['Compound'].sum() / t_min, 3)
        c_avg = round(dfx_1['Compound'].mean(), 3)

        with urllib.request.urlopen("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT") as url:
            dat = json.loads(url.read().decode())
            price = round(float(dat['price']), 2)

        p_logic = 0
        aux_logic = 0
        s_logic = 0

        span = 3  # Multiples of time_limit
        avg_win = 4
        expectancy = 1
        sample = '15Min'
        time_delay = 15  # Time delay in minutes for the prediction
        current_date = str(dfx_1.index.max())
        current_date_2 = dfx_1.index.max() + dt.timedelta(minutes=time_delay)

        if len(dfb) > span and span > 1:

            t_str = t_count + dfb.iloc[-span + 1:, 2].sum()
            p_str = p_min + dfb.iloc[-span + 1:, 4].sum()
            p_idx = (p_avg + dfb.iloc[-span + 1:, 5].sum()) / span
            n_str = n_min + dfb.iloc[-span + 1:, 6].sum()
            n_idx = (n_avg + dfb.iloc[-span + 1:, 7].sum()) / span
            c_str = c_min + dfb.iloc[-span + 1:, 8].sum()
            c_idx = (c_avg + dfb.iloc[-span + 1:, 9].sum()) / span
            pc_price = (price / dfb.iloc[-span, 10]) - 1


        elif len(dfb) > 1:

            t_str = t_count
            p_str = p_min
            p_idx = p_avg
            n_str = n_min
            n_idx = n_avg
            c_str = c_min
            c_idx = c_avg
            pc_price = (price / dfb.iloc[0, 10]) - 1

        if len(dfb) >= span * 2 and len(dfb) % span == 0:

            df1 = dfb.copy()
            df1 = df1.set_index(pd.DatetimeIndex(df1['Datetime']))
            df1 = df1.resample(f'{sample}').agg(
                {'Minutes': np.sum, 'Filtered Tweets': np.sum, 'Tweets per Min': np.mean, 'P.Strength': np.sum,
                 'N.Strength': np.sum, 'C.Strength': np.sum, 'P.Index': np.mean, 'N.Index': np.mean, 'S.Index': np.mean,
                 'Price': 'last'})
            pct_list = list(df1['Price'].pct_change())
            pct_list = pct_list[1:]
            pct_list.append(pc_price)
            df1['Sharpe Ratio'] = df1['Price'] * 0
            for x in range(1, len(df1)):
                df1['Sharpe Ratio'][x] = statistics.mean(pct_list[:x + 1]) / statistics.stdev(
                    pct_list[:x + 1]) * expectancy ** 0.5

            df1['Sharpe Ratio'] = df1['Sharpe Ratio'].fillna(0)

            if len(dfb) > avg_win * span:

                t_str_avg = df1.iloc[-avg_win:, 1].mean()
                t_str_dev = df1.iloc[-avg_win:, 1].std()
                p_str_avg = df1.iloc[-avg_win:, 3].mean()
                p_str_dev = df1.iloc[-avg_win:, 3].std()
                p_idx_avg = df1.iloc[-avg_win:, 6].mean()
                p_idx_dev = df1.iloc[-avg_win:, 6].std()
                n_str_avg = df1.iloc[-avg_win:, 4].mean()
                n_str_dev = df1.iloc[-avg_win:, 4].std()
                n_idx_avg = df1.iloc[-avg_win:, 7].mean()
                n_idx_dev = df1.iloc[-avg_win:, 7].std()
                c_str_avg = df1.iloc[-avg_win:, 5].mean()
                c_str_dev = df1.iloc[-avg_win:, 5].std()
                c_idx_avg = df1.iloc[-avg_win:, 8].mean()
                c_idx_dev = df1.iloc[-avg_win:, 8].std()
                sr_avg = df1.iloc[-avg_win:, 10].mean()
                sr_dev = df1.iloc[-avg_win:, 10].std()

            else:

                t_str_avg = df1.iloc[:, 1].mean()
                t_str_dev = df1.iloc[:, 1].std()
                p_str_avg = df1.iloc[:, 3].mean()
                p_str_dev = df1.iloc[:, 3].std()
                p_idx_avg = df1.iloc[:, 6].mean()
                p_idx_dev = df1.iloc[:, 6].std()
                n_str_avg = df1.iloc[:, 4].mean()
                n_str_dev = df1.iloc[:, 4].std()
                n_idx_avg = df1.iloc[:, 7].mean()
                n_idx_dev = df1.iloc[:, 7].std()
                c_str_avg = df1.iloc[:, 5].mean()
                c_str_dev = df1.iloc[:, 5].std()
                c_idx_avg = df1.iloc[:, 8].mean()
                c_idx_dev = df1.iloc[:, 8].std()
                sr_avg = df1.iloc[:, 10].mean()
                sr_dev = df1.iloc[:, 10].std()

            ts_w = 1
            ps_w = 1
            pi_w = 1
            ns_w = 1
            ni_w = 1
            cs_w = 1
            ci_w = 1
            sr_w = 1

            ts_logic = 0
            ps_logic = 0
            pi_logic = 0
            ns_logic = 0
            ni_logic = 0
            cs_logic = 0
            ci_logic = 0
            sr_logic = 0

            if t_str > t_str_avg + t_str_dev * ts_w:
                ts_logic = 1
            elif t_str < t_str_avg - t_str_dev * ts_w:
                ts_logic = -1

            if p_str > p_str_avg + p_str_dev * ps_w:
                ps_logic = 1
            elif p_str < p_str_avg - p_str_dev * ps_w:
                ps_logic = -1

            if p_idx > p_idx_avg + p_idx_dev * pi_w:
                pi_logic = 1
            elif p_idx < p_idx_avg - p_idx_dev * pi_w:
                pi_logic = -1

            if n_str > n_str_avg + n_str_dev * ns_w:
                ns_logic = -1
            elif n_str < n_str_avg - n_str_dev * ns_w:
                ns_logic = 1

            if n_idx > n_idx_avg + n_idx_dev * ni_w:
                ni_logic = -1
            elif n_idx < n_idx_avg - n_idx_dev * ni_w:
                ni_logic = 1

            if c_str > c_str_avg + c_str_dev * cs_w:
                cs_logic = 1
            elif c_str < c_str_avg - c_str_dev * cs_w:
                cs_logic = -1

            if c_idx > c_idx_avg + c_idx_dev * ci_w:
                ci_logic = 1
            elif c_idx < c_idx_avg - c_idx_dev * ci_w:
                ci_logic = -1

            if df1.iloc[-1, 10] > sr_avg + sr_dev * sr_w:
                sr_logic = 1
            elif df1.iloc[-1, 10] < sr_avg - sr_dev * sr_w:
                sr_logic = -1

            p_logic = ns_logic
            aux_logic = ts_logic + ps_logic + pi_logic + ni_logic + ci_logic

            pbs_strength = (n_str / (n_str_avg + n_str_dev * ns_w) - 1) * -100
            pss_strength = (n_str / (n_str_avg - n_str_dev * ns_w) - 1) * -100

            sec_list = [1, 3, 4, 5, 6, 7, 8, 10]
            sec_dict = {1: t_str, 3: p_str, 4: n_str, 5: c_str, 6: p_idx, 7: n_idx, 8: c_idx, 10: df1.iloc[-1, 10]}

            sbs_list = []
            sss_list = []
            super_list = []
            sbs = 0
            sss = 0

            # for x in sec_list:
            #
            #    x_avg = df1.iloc[-avg_win:,x].mean()
            #    x_dev = df1.iloc[-avg_win:,x].std()
            #    x_item = sec_dict.get(x)
            #    if x_item > x_avg + x_dev:
            #        if (x == 4 or x == 7):
            #            sss = (x_item/(x_avg + x_dev)-1) * -100
            #        else:
            #            sbs = (x_item/(x_avg + x_dev)-1) * 100
            #    elif x_item < x_avg - x_dev:
            #        if (x == 4 or x == 7):
            #            sbs = (x_item/(x_avg - x_dev)-1) * -100
            #        else:
            #            sss = (x_item/(x_avg - x_dev)-1) * 100
            #    else:
            #        sbs=0
            #        sss=0

            for x in sec_list:

                x_avg = df1.iloc[-avg_win:, x].mean()
                x_dev = df1.iloc[-avg_win:, x].std()
                x_item = sec_dict.get(x)
                sbs = 0
                sss = 0
                if x_item > x_avg:
                    if (x == 4 or x == 7):
                        sss = (x_item / x_avg - 1) * -100
                        sbs = 0
                    else:
                        sbs = (x_item / x_avg - 1) * 100
                        sss = 0
                elif x_item < x_avg:
                    if (x == 4 or x == 7):
                        sbs = (x_item / x_avg - 1) * -100
                        sss = 0
                    else:
                        sss = (x_item / x_avg - 1) * 100
                        sbs = 0

                sbs_list.append(sbs)
                sss_list.append(sss)
                s = sbs + sss

                super_list.append(s)

            sbs_strength = sum(sbs_list) / len(sbs_list)
            sss_strength = sum(sss_list) / len(sbs_list)

            if aux_logic > 0:
                s_logic = 1
            elif aux_logic < 0:
                s_logic = -1

            if p_logic + s_logic > 0:

                message = f'''
                Probable alza a las {current_date_2}
                Primary Signal Strength: {pbs_strength:.2f} %
                Secondary Signal Strength: {sbs_strength:.2f} %
                '''

                print(f'{message}')
                m_data = [{'Datetime': current_date, 'P.Signal': pbs_strength, 'S.Signal': sbs_strength}]
                dfc = dfc.append(m_data, ignore_index=True, sort=False)

            elif p_logic + s_logic < 0:

                message = f''' 
                Probable baja a las {current_date_2}
                Primary Signal Strength: {pss_strength:.2f} %
                Secondary Signal Strength: {sss_strength:.2f} %
                '''

                print(f'{message} \n')
                m_data = [{'Datetime': current_date, 'P.Signal': pss_strength, 'S.Signal': sss_strength}]
                dfc = dfc.append(m_data, ignore_index=True, sort=False)

            d_data = [{'Datetime': current_date, 'TS Signal': super_list[0], 'PS Signal': super_list[1],
                       'NS Signal': super_list[2],
                       'CS Signal': super_list[3], 'PI Signal': super_list[4], 'NI Signal': super_list[5],
                       'CI Signal': super_list[6],
                       'SR Signal': super_list[7], 'Price': 0}]

            dfd = dfd.append(d_data, ignore_index=True, sort=False)

        t_avg = round(t_count / t_min, 2)
        t_min = round(t_min, 2)

        data = [{'Datetime': current_date, 'Minutes': t_min, 'Filtered Tweets': t_count,
                 'Tweets per Min': t_avg, 'P.Strength': p_min, 'P.Index': p_avg, 'N.Strength': n_min,
                 'N.Index': n_avg, 'C.Strength': c_min, 'S.Index': c_avg, 'Price': price}]
        print(data)

        dfb = dfb.append(data, ignore_index=True, sort=False)

        return dfx_1, dfb, dfc, dfd


def hashtag_extract(x):
    hashtags = []

    # Loop over the words in the tweet

    for i in x:
        ht = re.findall(r"#(\w+)", i)
        hashtags.append(ht)
    return hashtags


def trending_analysis(df):
    HT_positive = []

    # extracting hashtags from positive tweetsHT_positive = hashtag_extract(df_tws['text'][df_tws['sent'] == 1])

    HT_positive = hashtag_extract(df['text'][df['Compound'] > 0.5])
    HT_negative = hashtag_extract(df['text'][df['Compound'] < 0])

    # unnesting list

    HT_positive = sum(HT_positive, [])
    HT_negative = sum(HT_negative, [])

    print(f'Positive Trending Topics: \n {HT_positive} \n')
    print(f'Negative Trending Topics: \n {HT_negative} \n')


def word_cloud(wd_list):
    stopwords = set(STOPWORDS)
    all_words = ' '.join([text for text in wd_list])
    wordcloud = WordCloud(
        background_color='white',
        stopwords=stopwords,
        width=1600,
        height=800,
        random_state=1,
        colormap='jet',
        max_words=80,
        max_font_size=200).generate(all_words)
    plt.figure(figsize=(12, 10))
    plt.axis('off')
    plt.imshow(wordcloud, interpolation="bilinear")


count = 0


def backup(save_min, dfb, dfc, dfd):
    global count
    current_time = time.time() - start_time
    checkpoint = count * save_min
    if int(current_time / 60) > checkpoint:
        csv_name = dt.datetime.now().strftime("%Y-%m-%d")
        dfb.to_csv(f'{csv_name}')
        dfc.to_csv(f'{csv_name} Predictions')
        dfd.to_csv(f'{csv_name} Signals')
        count += 1
        print('saved')


def plot(df):
    fig, ax = plt.subplots(figsize=(16, 5))

    if len(df) > 3 and len(df) < 24:

        ax.plot(df['Datetime'],
                df['NS Signal'],
                color='purple')

    elif len(df) > 24:

        ax.plot(df['Datetime'][-24:],
                df['NS Signal'][-24:],
                color='purple')

    ax.set(xlabel="Datetime",
           ylabel="Signal Strength %",
           title="Live Signal")
    plt.xticks(rotation=90)
    plt.show()


def backtest(df, df2, model='EWA', low_rsi=40, high_rsi=60):
    global long_context
    global short_context
    if len(df) > 14:

        df['Return'] = df['Price'].diff()

        df['RS+'] = df['Return'].copy()
        df.loc[df['Return'] < 0, 'RS+'] = 0
        df['RS-'] = df['Return'].copy()
        df.loc[df['Return'] > 0, 'RS-'] = 0

        df['EWA-RS'] = (df['RS+'].ewm(com=13, min_periods=14).mean()) / abs(
            df['RS-'].ewm(com=13, min_periods=14).mean())
        df['EWA-RSI'] = 100 - 100 / (1 + df['EWA-RS'])

        ewma = (df['Price'].ewm(9).mean().iloc[-3:] / df['Price'].ewm(21).mean().iloc[-3:]) - 1
        print(f'EWMA Max % = {ewma.max()}')
        print(f'EWMA Min % = {ewma.min()}')

        rsi = df[model + '-RSI']
        print(f'RSI Max= {rsi.iloc[-3:].max()}')
        print(f'RSI Min= {rsi.iloc[-3:].min()}')
        cp = df['Price'].iloc[-1]

        if df2.iloc[-1, 3] < df2.iloc[-2, 3] and rsi.iloc[-3:].max() > high_rsi and ewma.max() > 0.002 and not short_context:
            short_context = True
            long_context = False
            df2['Price'].iloc[-1] = cp

            print(f'Sold at {cp}')

        if df2.iloc[-1, 3] > df2.iloc[-2, 3] and rsi.iloc[
                                                 -3:].min() < low_rsi and ewma.min() < -0.002 and not long_context:
            short_context = True
            long_context = False
            df2['Price'].iloc[-1] = cp * (-1)
            print(f'Bought at {cp}')

    return df2

