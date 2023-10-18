

# Stream will stop when reaching an amount of tweets or when reaching an amount of time by checking variables in that order.
ta_mode = 'everyone'  # ta_mode defines where the sentiment comes from either by everyone or by a group of user defined in "influencers" variable.

output_file = "crypto.csv"
language = 'en'
since_min = 120 # Filters the last n minutes of tweets

# Stream will stop when reaching an amount of tweets or when reaching an amount of time by checking variables in that order.

tweet_count = 0  # Stops Stream at certain amount of tweets (Set to 0 to use time instead)
time_limit = 300  # Stops Stream at certain amount of time
save_min = 5  # Backups results to csv every n minutes

# Set Filters

filters = ['rt', 'retweet', 'follow', 'giveaway', 'tag ', 'give away', 'giving away', 'givingaway', 'free']

influencers = ("@NickSzabo4", "@nic__carter", "@CarpeNoctom", "@ToneVays", "@Melt_Dem","@CryptoHustle",
         "@MessariCrypto", "@TuurDemeester", "@aantonop","@CamiRusso", "@lopp", "@TimDraper",
         "@RyanSAdams", "@sassal0x", "@Rewkang", "@cryptoadvisory", "@DeFi_Dad", "@ChrisBlec", "@CharlieShrem",
         "@Arthur_0x", "@spencernoon", "@defipulse", "@evan_van_ness", "@defiprime", "@TrustlessState",
         "@APompliano", "@StaniKulechov", "@nanexcool", "@defikaren", "@econoar", "@LucasNuzzi", "@DefiantNews",
         "@VitalikButerin","@VladZamfir","@ethereumJoseph","@gavofyork", "@naval", "@tayvano_", "@gavinandresen",
         "@simondrl", "@cburniske", "@jwolpert", "@iam_preethi","@rogerkver", "@brockpierce", "@SatoshiLite",
         "@officialmcafee","@JihanWu", "@AriDavidPaul", "@TuurDemeester", "@jimmysong","@mikojava", "@durov",
         "@peterktodd", "@adam3us", "@wheatpond", "@VinnyLingham", "@ErikVoorhees", "@barrysilbert", "@TimDraper",
         "@brian_armstrong", "@koreanjewcrypto", "@MichaelSuppo", "@DiaryofaMadeMan","@aradchenko1","@pmarca",
         "@CryptoYoda1338","@ToneVays", "@CremeDeLaCrypto", "@TheCryptoDog","@laurashin","@cryptoSqueeze",
         "@iamjosephyoung","@ForbesCrypto","@Crypto_Bitlord", "@woonomic", "@parabolictrav", "@Melt_Dem", "@haydentiff"
         "@CryptoDonAlt", "@Fisher85M","@jonmatonis", "@Beastlyorion", "@moonshilla", "@ProfFaustus", "@kyletorpey",
         "@TuurDemeester", "@pierre_rochard", "@francispouliot_", "@AriannaSimpson", "@ArminVanBitcoin", "@thecryptomonk",
         "@BitcoinByte","@Xentagz", "@CryptoTrooper_","@SDLerner")

keywords = ["cryptocurrency", "digital asset", "crypto", "btc", "bitcoin", "satoshi", "bch", "eth", "ethereum"]

long_context = False
short_context = False
