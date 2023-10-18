# X-Sentiment-Analysis-
X/Twitter Sentiment Analysis and backtesting for trading

1- Introduction

This tool was designed to execute a real-time sentiment analysis based on X/Twitter-filtered comments and then use this information along with basic technical indicators to print buy/sell signals in real time and plot them on-screen.

2- Setup 

In order to execute the tool we need to first configure the SA_Keys.py file with the corresponding API_keys from X/Twitter account and set the stream parameters in SA_Config.py. Once this is done the program is executed in SA_Main.py.

3- Considerations

This tool was designed when Twitter streaming services were free, currently, this service is no longer free so in order to see if this still works you will need APIs from an X-paid account. Since I don't have one I was only able to update the program to a certain point and it may need further debugging and testing that can only be done with these keys. Bear in mind that even with the correct paid-keys the program might present errors due to outdated endpoints and deprecated methods.
