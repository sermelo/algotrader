> **Source:** Cohen, *Mathematics* 10(18) (2022) 3302, <https://doi.org/10.3390/math10183302>  
> **Licence:** <https://creativecommons.org/licenses/by/4.0/> (copyright the author). Unmodified text, converted from the publisher PDF to Markdown with pymupdf4llm 0.3.4; figures omitted. Conversion garbles equations and some tables: check the original for anything that matters.

# **_mathematics_**

_Review_
## **Algorithmic Trading and Financial Forecasting Using Advanced** **Artificial Intelligence Methodologies**

**Gil Cohen**

Department of Management, Western Galilee Academic College, Acre 2412101, Israel; gilc@wgalil.ac.il

**Abstract:** Artificial Intelligence (AI) has been recently recognized as an essential aid for human
traders. The advantages of the AI systems over human traders are that they can analyze an extensive
data set from different sources in a fraction of a second and perform actual high-frequency trading
(HFT) that can take advantage of market anomalies and price differences. This paper reviews the
most important papers published in recent years that use the most advanced techniques to forecast
financial asset trends and answer the question of whether those techniques can be used to successfully
trade the complex financial markets. All systems use deep learning (DL) and machine learning (ML)
protocols to explore nonobvious correlations and phenomena that influence the probability of trading
success. Their predictions are based on linear or nonlinear models often combined with social media
investors’ sentiment derivations or pattern recognitions. Most of the reviewed papers have proven
the successful ability of their developed system to trade the financial markets.

**Keywords:** algorithmic trading; fuzzy systems; LSTM; forecasting; Artificial Intelligence

**MSC:** O3D32; 68Q30

**Citation:** Cohen, G. Algorithmic

Trading and Financial Forecasting

Using Advanced Artificial

Intelligence Methodologies.

_Mathematics_ **2022**, _10_, 3302. [https://](https://doi.org/10.3390/math10183302)

[doi.org/10.3390/math10183302](https://doi.org/10.3390/math10183302)

Academic Editor: Maria C. Mariani

Received: 13 August 2022

Accepted: 9 September 2022

Published: 12 September 2022

**Publisher’s Note:** MDPI stays neutral

with regard to jurisdictional claims in

published maps and institutional affil

iations.

**Copyright:** © 2022 by the author.

Licensee MDPI, Basel, Switzerland.

This article is an open access article

distributed under the terms and

conditions of the Creative Commons

Attribution (CC BY) license [(https://](https://creativecommons.org/licenses/by/4.0/)

[creativecommons.org/licenses/by/](https://creativecommons.org/licenses/by/4.0/)

4.0/).

**1.** **Introduction**

Trading in the financial markets is complex and chaotic because of the large number
of influential factors from the economic and psychological surroundings. The field is also
characterized by the high speed of information inflows that affects trade profitability within
seconds. For these reasons, researchers and practitioners have tried to use Artificial Intelligence (AI) that can help the individual investor make trading decisions or automatically
conduct the entire trading process. The aim of the following paper is to review the most
recent advances in algorithmic trading systems that are based on AI and to evaluate their
future evolutions. Moreover, unlike other papers that concentrate on a single methodology or data set, in this paper we took a more holistic approach that integrates different
methodology or data sources into a single decision-making process aimed at detecting
financial asset price trend shifts. In our view, the analysis of combined information that
is collected from different sources can better forecast financial asset price movement. The
data are collected from technical and fundamental analysis and investor sentiment derived
from social media. Moreover, we report research on different financial assets such as stocks,
ETFs and cryptocurrencies, and precious metals on different trading time frames that can
be used by a wide range of traders, including swing and intraday.

A typical algorithmic trading system can accurately analyze various data sets from
different sources, produces buy/sell signals through machine learning (ML) or deep learning (DL) artificial neural networks (ANN), and can transact a large number of orders with
high-frequency trading (HFT) in a fraction of a second. For that reason, many institutional
traders are developing trading systems, which allow them to perform a large number
of financial transactions in a short period. HFT algorithm can also be useful to perform
arbitrage trading that takes advantage of the existence of different prices for the same commodity. For such strategies, HFT firms need fluent non delayed data and will be willing

_Mathematics_ **2022**, _10_, 3302. [https://doi.org/10.3390/math10183302](https://doi.org/10.3390/math10183302) [https://www.mdpi.com/journal/mathematics](https://www.mdpi.com/journal/mathematics)

_Mathematics_ **2022**, _10_, 3302 2 of 13

to pay a lot of money to acquire a fast and fluent data stream that will provide them with
reliable data before their competitors. According to authors of [1] traders often hold their
positions for short periods of time and profit from tiny gains in each trade. Furthermore,
the volume of HFT trading has been growing rapidly since the beginning of the 21st century
to approximately 70% at its peak. Ref. [2] offered agent-based simulation that will be able
to replicate several characteristics of financial markets including the presence of extreme
price events, long memory in order flow, the autocorrelation of returns, concave price
impact, and clustered volatility. Ref. [3] examined the effect of high-frequency trading
on market quality by examining limit orders. They argued that HFT traders impose a
welfare externality by gathering the most profitable limit orders and creating order book
imbalance. They concluded that HFT traders demand liquidity from the thin side of the
order book and supply liquidity on the thick side. This strategy in their view is more
pronounced during the volatile periods when trading speed increases. Ref. [4] argued
that the use of HFT strategies combined with machine learning methods has significantly
improved the capability of financial time series forecasts. They proposed an effective and
robust automatic HFT grid trading system for the FOREX markets that reduced trade
drawdown. Ref. [5] investigated the predictability of various machine learning models for
high-frequency trading of Bitcoin and concluded that they had long memory traits, high
levels of stochasticity, and topological complexity. They showed that the superiority of the
artificial neural networks is sourced by the parallel processing methods that emulate human
decision-making in the presence of underlying nonlinear input–output relationships in
noisy signal environments and that the Bayesian regularization renders an outstanding
accuracy in forecasting.

AI Trading systems are usually based on several of the following data sources and
methodologies: 1. Technical analysis, which is a set of strategies that are derived from past
behavior of a financial assets price and volumes of trade. This methodology generates
trading signals of different financial assets, such as stocks, currencies, EFTs (EFT = Exchange
Trade Fund), and commodities. 2. Fundamental analysis which evaluates the impact
of corporate economic news on stock prices. 3. Investors sentiment implies investors’
emotions which are gathered from social media. This paper will review the most advanced
research papers in the field of algorithmic trading that combine the described data sources
and methodologies into an integrated system. Moreover, the paper will show which
methodology combinations helped improve trading results and which proved nonfertile
in improving trading performances. Moreover, we report the results of recent research
that combines technical analysis and pattern recognition with more traditional forecasting
methods such as linear regression.

Technical analysis has been developed enormously during past decades and gained
popularity among novice and professional trades because of its relative simplicity and
ease of performance. Furthermore, the AI revolution has emphasized the advantage of
such analysis when it is conducted by machines rather than by human traders. Machines
can wait for market entry or exit for a long period without getting tired and they are not
subject to human bias and emotions that sometimes lead human traders to bad investment
decisions. Ref. [6] found that whether algorithm traders earn more profit than human
traders depends on the market environment and the fundamental value of a financial asset.
Ref. [7] found evidence of an inverse relationship between algorithmic trading efficiency
and trade size.

DL methodology is an algorithm composed of multiple ANN layers. The procedure
sets multiple feature layers to obtain higher-dimensional classification and it is being used
to analyze an unbounded number of layers of information. Learning could be supervised,
semi-supervised, or unsupervised. This methodology can be used to identify correlations
between asset returns by examining an infinite number of past data and producing future
returns predictions according to the occurrence of identified past conditions. For example,
DL can integrate past commodity prices with interest rates, currencies, inflation data,

_Mathematics_ **2022**, _10_, 3302 3 of 13

and returns of specific stocks or ETFs to produce future return probabilities of any of the
observed assets.

Another field of knowledge that gains popularity in recent years is the use of AI to track
investment emotions towards a particular financial asset through internet word tracking
in general with emphasis on social networks. Those emotional recognitions are gathered
and filtered and then are used to produce a ratio of positive divided by negative emotions
that are generated in the decision-making system as another source of information that can
strengthen or weaken other inputs before predicting a financial asset’s price direction.

Advanced algorithmic trading systems can perform actual trading according to AI and
report the trading results including, for example, price slippage and brokers fees. Moreover,
those systems often report three performance measurements: Net Profit (NP), Percent
of Profitable trades of all trades (PP), the Profit Factor (PF), and Maximum Drawdown
(MDD). NP is the total net profit generated by the trading system. The PP is the percentage
of winning trades of all trades generated by the system and the PF is the gross profits
divided by gross losses and MDD measures the maximum loss generated by a single trade.
Each investor can pick the trading algorithm that produces the best back-testing trading
results according to his/her risk preferences. A more risk-averse investor will prefer an
algorithmic setup that produced low MDD while a less risk-averse investor may prefer an
algorithm that produces the highest NP or PF. The purpose of this paper is to review the
most advanced methodologies that are used for AI financial market analysis and trading.

Trading systems that use AI have been developed dramatically in recent years. Those
systems seek to improve predictions of financial asset price trends and as a result of that
improve trading performances. The evolutions of those systems led them to seek as many
inputs as possible derived from a variety of sources that can contribute to that task. A
successful trading system integrates a variety of information analyzed with a predesigned
system that understands the importance of each piece of information and its use in the
prediction process. Such a procedure is complicated and challenging and demands careful
algorithm selection and design.

Algorithmic trading has its merits; however, they are also associated with risks and
costs that are unique to computational ML systems. For example, systems that are based on
algorithms may interpret a momentarily trading disruption caused by an external problem
as a market crash and enhance the price fall causing major losses to investors. Another
example is the system can interpret one extreme price wrongly generated by a single
trader as a signal of an upcoming trend shift in the financial assets price. Those examples
demonstrate that the system’s need for a fast reaction to market new information may
result in a wrong interpretation of the news and substantial losses. Ref. [8] stated that risk
arises from the choice of the algorithms, bias, and non-representative data sets and from
human decisions that are based on their AI interpretations. They added that the reduction
in risks requires a vigilant division of work between AI and humans. Costs of trading may
also be a problem to algorithmic trading systems that usually perform many trades in a
short period of time. That system must have a rebate arrangement that reduces its costs in
order to maximize profits.

**2.** **Material and Methods**

This paper reviews recent techniques and methodologies utilized to construct algorithmic trading systems using AI and DL capabilities. The papers were chosen according to
their originality and importance to the field. The publication date and the journal they were
published in were also taken into consideration. The papers that are cited in this study
use different methodologies to forecast financial asset trends. They all have something
in common: they use past information to predict the future and show that DL and ML
can be used to process that data to produce successful trading performances. Combining
traditional methods such as technical analysis, pattern recognition, and investor sentiment
analysis with advanced DL algorithm have been proven to be able to improve financial
forecasting and trading profitability. Moreover, integrating investors’ psychology and

_Mathematics_ **2022**, _10_, 3302 4 of 13

biases into trading systems has become more and more popular in recent years following
the iconic work of Kahneman and Tversky’s prospect theory (Ref. [9]).

This paper discusses the fuzzy logic that guides algorithmic trading systems. That logic
analyses data from different sources such as past price movements of financial assets and
correlations, volumes of trade, economic surroundings, firms’ fundamental reporting, and
investors’ sentiments. Such a vast amount of information sometimes diverts or delays the
system from achieving the correct and on-time trading decisions. In order to cope with these
challenges, an optimal number of rules and a “forget” mechanism that excretes unnecessary
information, should be used. A recent publication, for example, found that adding more
indicators and trading rules does not necessarily improve the trading results [10]. They
documented that the optimal number of technical indicators for trading daily bars of major
cryptocurrencies is one or two.

The paper structure is as follows: Chapter 3 discusses artificial neural networks use in
financial assets price predictions, including Section 3.1 on support vector machine, Section
on 3.2 long short-term memory, and Section on 3.3 fuzzy systems. Chapter 4 discusses
the important role of investors’ sentiment and social media impact on financial asset price
predictions and how these information pieces are integrated into autonomous trading
systems. Recent publications have pointed out that social media sentiment measurement
can contribute information that other sources of row data cannot and therefore they can
be used to improve price predictions and trading performance. Chapter 5 discusses
pattern recognitions and their contributions to financial assets price forecasting and trading.
These patterns have been used for many years by professional traders, and they are often
being integrated into prediction systems in finance. Chapter 6 summarizes and concludes
this paper.

**3.** **Artificial Neural Network (ANN)**

Artificial neural networks (ANN) are known to be able to provide an abnormal return
by using technical indicators as predictors in stock markets [11]. Ref. [12] achieved excess
returns by using deep learning techniques and technical indicators to analyze open-highlow-close prices and volume in the Korean stock market. The ANN, as a DL technique is
used to recognize patterns or images by imitating the visual processing of living organisms.
Ref. [13] predicted cryptocurrency prices using a stochastic model that induces layer-wise
randomness into the observed feature activations of NN. They achieved in their bestcase experiment, only 5% improvement in Mean Absolute Percentage Error (MAPE) with
23 features of a 7-day window size. Ref. [14] offered cryptocurrency prices forecasting
methodology using a DL algorithm and blockchain information. They concluded that
prediction performance is high for both currencies when the blockchain data are used
together with Bitcoin and Ethereum prices. Moreover, they documented that the Ethereum
currency has the highest percentage of prediction accuracy and the lowest error rate
compared to Bitcoin. NN uses learning filters that are generated in each of the network
layers to detect a predetermined signal at any given layer and to send real-time information
to the system that performs automated trading decisions.

_3.1._ _Support Vector Machine (SVM)_

Support vector machine (SVM) is a supervised learning method aimed to predict and
classify items using linear or nonlinear processes [15]. The process allows error within the
training data resulting in a significantly reduced error in the tested data. Ref. [16] combined
a cumulative auto-regressive moving average with a least squares SVM model to make basic
predictions for the stock market. They concluded that their model is more suitable for stock
price forecasting than the single forecasting model. Ref. [17] tested the predictability of the
NIKKEI 225 index weekly movement direction with the SVM forecasting model and proved
that SVM outperforms other classification methods. Furthermore, they offered an improved
model that combines SVM with the other classification methods. Ref. [18] used the SVM
algorithm to make automated transaction decisions on the Forex market and proved the

_Mathematics_ **2022**, _10_, 3302 5 of 13

system’s contribution to successful trading. Ref. [19] examined the predictability of the
12 cryptocurrency prices using SVM and concluded at the daily or minute level frequencies,
ML classification algorithms reach about 55–65% of predictive accuracy. Ref. [20] used SVM
kernel functions to predict the price directions of cryptocurrency and foreign exchange.
They showed that the Radial Basis Gaussian (RBG) SVM forecasting model outperformed
other SVM-kernel models.

SVM-based systems are widely used to predict financial asset price trends. Those
systems use different models that filter past financial data and produce predictions. They
all assume that future price trends can be forecasted by analyzing past correlations between financial assets and that economic conditions are repetitive. Such assumptions are
correct under normal market conditions; however, they prove infertile under “Black Swan”
conditions that surprise the market from time to time.

_3.2._ _Long Short-Term Memory (LSTM)_

LSTM is an artificial recurrent neural network (RNN) used in the fields of AI and
DL. The procedure can process sequences of data simultaneously and it is useful for
classifying, processing, and making predictions on time serious data, making it an ideal
tool for financial assets price predictions. LSTM includes a cell, input and output gates, and
a forget gate (see Figure 1). The cell remembers values while the other three gates control
the information flow into and out of the cell.

**Figure 1.** LSTM System. Sources: Refs. [21,22] (accessed on 8 March 2022).

_Ct_ in Figure 1 is the Cell in which the main processes are conducted, and _it_, _ot_ and
_ft_ are the input, output, and the forget gates. The mathematical process is described in
Equations (1)–(5).

_ft_ = _σg_

_Wf xt_ + _U_ _f ct−_ 1 + _b_ _f_

(1)

(2)

_it_ = _σg_

_Wf xt_ + _Uict−_ 1 + _bi_

_ot_ = _σg_ ( _Woxt_ + _Uoct−_ 1 + _bo_ ) (3)

_ct_ = _ft_ _o ct−_ 1 + _it_ _o σc_ ( _Wcxt_ + _bc_ ) (4)

_ht_ = _ot_ _o σh_ ( _ct_ ) (5)

_Mathematics_ **2022**, _10_, 3302 6 of 13

where: _ft_ _∈_ (0, 1) _[h]_ is the forget gate activiation vector, _f it_ _∈_ (0, 1) _[h]_ is the input gte
activiation vector, _ot_ _∈_ (0, 1) _[h]_ is the output gate activiation vector, _ct_ is the cell vector,
_ht_ _∈_ ( _−_ 1, 1) _[h]_ is the output vector, _σg_ = sigmoid function, _σc_, _σh_ = hyperbolic tangent
functions, _W_, _U_ is the weight matrix, and _b_ is the bias vector that are learned by training.

The major difference between LSTMs over traditional ANN is that they can learn
selectively by remembering and forgetting the required historical data. This can be very
useful for the financial assets trading system by setting a specific range of data that the
designer wants the algorithm to perform predictions while at the same time, deciding upon
the range and scope of data that the system should forget in order not to be destructed by
irrelevant data. Ref. [23] ensembled an LSTM model that uses many technical indicators
as the network inputs for predicting intraday stock movements of U.S. large-cap stocks.
They concluded that the model performed better than benchmark models with lasso and
ridge logistic classifiers or equally weighted ensembles. Ref. [24] used LSTM networks
with technical analysis indicators to predict future stock prices. Results showed an average
of 55.9% accuracy in predicting stock price trend shifts. Ref. [25] compared deep learning
methods such as a deep neural network (DNN), a convolutional NN, and a deep residual
network, for Bitcoin price prediction. They showed that DNN-based models were the best
for price trend prediction of Bitcoin. Moreover, they found that the classification model was
more effective than regression models for algorithmic trading. Ref. [26] tested forecasting
models using daily cryptocurrency prices. For each cryptocurrency, they used different
LSTM models and concluded that the model that is based on 50 days of data performed the
best since it has the capability to capture long-term dependencies. Ref. [27] used LSTM for
swing traders to predict future stock values. Their system reported the predicted values of
the company stock price in the Indian stock market for 30 days. Their system was based on
technical indicators such as Support and Resistance levels, Fibonacci retracement levels,
MFI, RSI, and the MACD (Support and Resistance levels, Fibonacci retracement levels,
MFI, RSI, and the MACD are technical indicators). The results achieved by the proposed
model establish the efficacy of the proposed technique. Ref. [28] employs random forests
and LSTM networks as training methodologies to forecast the movement of the S&P500
stocks from January 1993 until December 2018 for intraday trading. On each trading day,
they bought 10 stocks with the highest probability to outperform the market and sell
short 10 stocks with the lowest probability to do it. They showed that a multi-feature
setting provided a daily return of 0.64% using LSTM networks. Ref. [29] conducted their
research on China’s stock market; their data included opening price, closing price, lowest
price, highest price, and daily trading volume of stocks. Their results showed that the
LSTM neural network model can predict stock prices with limitations such as time lag
of prediction. Moreover, to predict stock price trends, the model should explore internal
rules through the selective memory advanced deep learning function of the LSTM neural
network model.

The advantage of an LSTM in financial price forecasting is its ability to forget information that the system designer evaluates their relevance and therefore should be ignored
after some time to enable correct predictions. This is a delicate task that can damage the
system’s prediction ability since the financial markets are very dynamic and the relevancy
of any specific data pieces to predictions may change over time.

_3.3._ _Fuzzy Systems_

A fuzzy system is a distinguished formation characterized as a set of interacting
components with a well-defined structure. The uniqueness of fuzzy techniques is that
they can process information, whereas classical theory or binary logic is impossible. fuzzy
systems can model complex and often vague real problems. The inputs of fuzzy systems
are sets of data with some degree of classified membership between 0 and 1 representing
the strength of belonging to a specific fuzzy set. Fuzzy logic is a method of reasoning that
resembles how humans perform decision-making. The fuzzy logic process consists of four
main parts, as demonstrated in Figure 2.

_Mathematics_ **2022**, _10_, 3302 7 of 13

**Figure 2.** Fuzzy Logic Process. Source: [https://www.edureka.co/blog/fuzzy-logic-ai.(accessed on](https://www.edureka.co/blog/fuzzy-logic-ai.(accessed)
10 July 2022).

Rules are the conditions offered by experts behind the algorithm. The Fuzzifier
converts inputs into fuzzy classification sets. The Intelligence determines the degree of
match between fuzzy inputs and the set of rules. At this stage, the system can also decide
on the set of rules that do not improve prediction and should be dismissed. The final
stage is Defuzzification in which the system converts the fuzzy sets into values that can be
translated into action orders such as buying or selling a financial asset.

Fuzzy systems are often used in financial forecasting when traditional models have
limited prediction power because of nonlinearity and nonstationary often characterizing
financial phenomena. Ref. [30] used technical analysis to construct a fuzzy system that
is able to perform medium-term optimal financial asset allocation. They showed that
their model has outperformed the Buy and Hold (B&H) strategy between 1997 and 2012
for ASE (ASE = Athens Stock Exchange). Refs. [31,32] used a clustering genetic fuzzy
system to predict daily stock prices. They used steps models that included stepwise
regression analysis, data clustering, and a genetic fuzzy system for stock price prediction
and concluded that their model has better forecasting accuracy of short-term stock price
than alternative models. Ref. [33] studied the ability of a multi-objective fuzzy system to
predict a bull signal or a bear signal in the cryptocurrency market. The fuzzy sets were
partitioned as very low, low, medium, high, and very high, later to be used to signal
whether a cryptocurrency is bearish or bullish on a specific day. They reported that the
system yields a 53% accuracy with 25 days rule length. Ref. [34] proposed a clusteringbased adaptive neuro-fuzzy system for predicting apple stock daily price. He used four
technical indicators in his study: 1 and 2 weeks simple moving average, 14 days disparity
(the disparity index measures the relative position of an asset’s most recent closing price
to a selected moving average and reports the value as a percentage) and Larry Williams
percent range (the Williams percent range is a momentum indicator that moves between 0
and _−_ 100 and measures oversold and overbought levels). The simulation results indicated
that the average subtractive clustering-based neuro-fuzzy networks were better than other
networks. Ref. [35] boosted the profitability of technical analysis for currency trading
using a multivariate fuzzy logic. Their approach yielded consistently higher Sharp ratios
(Sharp ratio measures the risk return ratio) for all investigated currencies. Ref. [36] suggests
evolving possibilistic fuzzy modeling to forecast realized volatility with jumps. Adding or
removing clusters was allowed using statistical distance-like criteria to update the model.
The data consisted of global major market indexes (S&P500 and Nasdaq, FTSE, DAX, IBEX,
and Ibovespa). Their results show that the possibilistic fuzzy model is highly efficient to
model index volatility with jumps in terms of forecasting accuracy. Ref. [37] proposed a
hybrid neuro-fuzzy controller to forecast the direction of the daily price of Bitcoin. Their

_Mathematics_ **2022**, _10_, 3302 8 of 13

system investment returns were 71.21% higher than the B&H strategy. Ref. [38] used fuzzy
if–then rules for their stock trading system. They used three linguistic variables as inputs
for the rules: view from the expert, earning-per-share, and price-to-earnings ratio. They
concluded that using this fuzzy logic can improve trading results. Ref. [39] tested whether
the occurrence of technical patterns along with a membership value is able to signal the
future returns of stocks. They matched 1450 U.S. companies with control companies on
the basis of market value and previous years’ returns and found that stocks with certain
technical patterns can generate abnormal returns after pattern occurrence for up to 120 days.
Ref. [40] combined technical analysis and fundamental analysis within a fuzzy system that
utilizes 677 stocks from the Indonesian stock market exchange. They concluded that the
Fuzzy technique combined with three candlesticks momentum entry buy/sell signals can
make trading more rewarding. The use of any trading rule depends on the time frame
preferences of the individual trader. Swing traders for example analyze daily and weekly
bars (a trading bar contains information about the opening/closing prices along with
the high/low levels of a specific time frame) while day traders analyze shorter periods
such as an hour or even a few minutes bars and base their decision upon a short period’s
momentums. Ref. [41] examined the intraday algorithmic system for trading precious
metals futures and found that an RSI-based system produced above the B&H returns of
326.3%, 106.2%, 63.7%, and 22.4%, for palladium, gold, silver, and platinum, respectively,
for the examined period. They also recommended that 60 min bars should be used for
trading precious metal futures. These results can be useful for intraday traders both manual
and algorithmic by implementing the most efficient trading rules and time frames for each
of the precious metals futures.

The advantage of the fuzzy systems in predicting financial asset price trends is their
relative simplicity and their ability to integrate different sources of information. The
problem with such systems is their inability to adjust to changing financial conditions and
relaxing predetermined rules that should be changed accordingly.

**4.** **Investors Sentiments and Social Media**

Social media has been growing in popularity among financial analysts because of its
ability to examine human sentiments towards various issues and not just raw data. The
idea is that “herd behavior” that exists in many life aspects also characterizes investment
behavior [42,43]. The use of investor sentiment analysis to predict stock return in the
financial literature sometimes appears as sole predicting information or as an addition to
other market data sources. Ref. [44] investigated the impact of social media on various
global financial markets using a firm’s Twitter sentiment and found that Twitter activity and
investor’s sentiment are associated with trading volume and returns. Ref. [45] predicted
changes in Bitcoin and Ethereum prices using Twitter and Google Trends data. They found
that tweet volume is a good predictor of price direction rather than tweet sentiment. They
formed a linear model that uses tweets and Google trends to accurately predict the direction
of price changes of cryptocurrencies.

Ref. [46] based on a large dataset of 84 million tweets and 8 years of stock data
for 407 companies from the S&P 500 index, tested whether social media has differential
impacts on stock performance. They found that while consumers’ negative sentiment
significantly impacts stock prices, consumers’ positive sentiment does not play a significant
role in stock performance. Ref. [47] argued that social media simultaneously encourages
and biases trading decisions creating a new class of self-directed online traders. Ref. [48]
combined indicators of technical analysis and investor sentiment derived from news articles
to construct a predictive model that can forecast trends of the twenty most capitalized
companies listed in the NASDAQ100 index. Ref. [49] argued that cryptocurrency price
fluctuations depend heavily on web search analytics tools such as Google Trends and on
social media sentiment. They also found that Twitter sentiments tend to be positive for
cryptocurrencies even if their prices go down.

_Mathematics_ **2022**, _10_, 3302 9 of 13

Ref. [50] applied several NN models to stock market opinions posted in twits and documented that the deep learning model can be used effectively for financial sentiment analysis
and a that the convolutional neural network is the best model to predict the sentiment out
of stock twit’s dataset. Ref. [51] tried to predict Bitcoin’s trends by extracting keywords
from Bitcoin-related comments posted on online forums. They claimed that user opinions
on online forums are conducive to understanding a range of cryptocurrency price trends.
Ref. [52] proposed the usage of ML tools and social media data for predicting the price
movement of Bitcoin, Ethereum, Ripple, and Litecoin. Their results indicate that machine
learning and sentiment analysis can predict cryptocurrency market trends. Moreover, they
concluded that Twitter data by itself can be used to predict certain cryptocurrencies’ prices.
Ref. [53] integrated sentiment analysis into a machine learning method and demonstrated
that the forecasting accuracy of the direction of the SSE50 (SSE50 is an index representing
50 top companies listed on the Shanghai Stock exchange) index is 89.93% with a rise of
18.6% after introducing sentiment variables. Their findings also imply that sentiment
probably contains precious information about the asset’s fundamental values. Ref. [54]
used text mining technology to quantify the social media opinions on stock-related news
compiled into logistic regression proposing an improved prediction model.

Social media and investors’ sentiments have been important in recent years as additions to financial price prediction processes. The understanding that row data alone has
limited prediction power is widely accepted now by researchers. However, investors’ mood
is easily altered by false rumors and fake news and therefore can be sometimes unreliable
and should be carefully analyzed. This is especially true for young or inexperienced investors who rely heavily on social media to shape their personal views. In order to reduce
that phenomenon and still use social media as a source of information, the system designer
should verify investors’ sentiments from various outlets before integrating them into the
trading system.

**5.** **Patterns Recognitions**

One of the most promising areas of AI is pattern recognition which is a DL method
applied to analyze pictures and photos in a variety of fields such as medicine, biology, and
chemistry. In recent years, this methodology has gained popularity among designers of
algorithmic trading systems and market forecasters. Years of trading experience have led
to the conclusion that certain price patterns are repeated in different financial markets and
financial assets. Those patterns are sourced by human behavior and biases. Since some
AI trading algorithms imitate human behavior and their intelligence is based on human
knowledge and experience, the pattern’s repeatability is getting stronger. The most basic
patterns recognition is “candlesticks” (candlestick is a formation representing the open,
close, high, and low price of a financial asset in a predetermined time frame) formations
and “Support and Resistance” levels for example [55,56]. Ref. [57] examined the ability
of “candlestick” patterns to predict Bitcoin’s trends shifts and found that the “Engulfing”
pattern can timely produce correct predictions of Bitcoin price shifts resulting in a 3.54 profit
factor that indicates that profits were 3.54 times the losses of the algorithmic trading system
during the beginning of 2012 till the end of July 2020. The result of this study can be used in
practice both by human traders and trading systems. A human trader can identify the daily
“Engulfing” buy/sell signals simply by examining the chart or using pattern recognition

systems and then performing the trading order manually. An automated trading system
can identify the pattern and produce trading orders accordingly.

Other researchers have tried to characterize financial asset price movement by Linear
Regression, Box formations, and dynamic channels such as Bollinger Bands (BB) and
Keltner channels (Keltner channels are volatility-based bands invented by Chester Keltner
that can aid in determining the direction of a trend). According to this trading strategy, a
breakout of the form implies the strength of the asset’s price momentum and can be used
to achieve abnormal returns. Ref. [58] used ML to design a trading system that is based on
three methodologies: Bollinger Bands (BB), Linear Regression, and Darvas boxes (Darvas

_Mathematics_ **2022**, _10_, 3302 10 of 13

boxes are box formations developed by Nicolas Darvas). The data in this study consist of
20 years of daily price data of five precious future metals: gold, silver, copper, platinum, and
palladium. Results show that linear regression is the best technique for forecasting silver
and gold prices, while the Bollinger Band technique best fits palladium prices. Ref. [59]
proposed an algorithmic system that is based on three fuzzy logic controllers such as
candlestick patterns and BB which were used for producing buy, hold, and sell signals.
Ref. [60] traded stocks of Taiwan 50 using BB methodology. Results reveal that investors
might beat the market by entering a long position when the price touches the lower band
and a short position when it touches the upper band.

Pattern recognition is traditionally used in forecasting systems in a variety of fields. In
finance predictions, they use the human psychological ability to examine chaotic environments with structures and formations. The assumption that leads systems designers to use
those patterns is that those patterns that appeared in the past will also appear in the future.
The problem with such an assumption is that patterns may change in their frequency of
appearance over time, and they differ from one financial asset to another.

**6.** **Summary and Conclusions**

This paper reviewed the most advanced systems that were designed to forecast and
trade financial assets. The paper’s main contribution is that it covers the most important
methodologies that are integrated into algorithmic trading systems. It also points to the
advantages and disadvantages of each methodology. The financial markets are complex
and dynamic, making forecasting a very difficult task. However, this paper shows that
predictions can be formed using AI and careful design producing profits that outperform
the B&H returns.

In recent years, researchers, institutional traders, and funders have become aware of
the advantages of using ANN for designing and operating trading systems. The advantages
that are inherent in such a system over human traders are enormous. An ANN system can
analyze a large amount of data from different sources in a fraction of a section. Moreover, it
can use fuzzy logic that imitates human behavior without the biases involved with human
trading. DL procedures can vaguely explore, though not intuitively, the correlation between
assets value and market economic conditions and use that for the price prediction of stocks,
ETFs currencies, and commodities. Algorithms can not only make accurate predictions
and perform real trading, they can also report various back-testing results that may suit
different risk-averse levels of investors. SVM can gather information from various sources
from financial exchanges, global economies, and social media. The latter source of information was documented to improve the system’s prediction accuracy. Another important
contribution of AI to trading is pattern recognition and breakout signaling. In this area of
expertise, the system is set to recognize well-known patterns and to decide accordingly
when to enter or exit a trade. Candlestick patterns, boxes, and channel formations are
found to be useful in predicting the price trend of various financial assets. Trading systems
seek to improve predictions of financial asset price trends and as a result of that to improve
trading performances. The evolutions of those systems led them to seek as many relevant
inputs as possible derived from a variety of sources that can contribute to that task. A
successful trading system integrates a variety of information analyzed with a predesigned
system that understands the importance of each piece of information and its use in the
prediction process. Such a procedure is complicated and challenging and demands careful
algorithm selection and design. An important practical implementation of this study is
that each financial asset behaves differently and therefore should be traded according to a
different set of rules and methodology. Moreover, the system designers must consider that
market conditions change frequently and change their systems accordingly.

The paper covered various methodologies that guide designers of algorithmic trading systems. Most of the cited papers have proven fertile in achieving the above index
profitability. Those suggested techniques must be modified in real life to a specific market condition and to a specific financial asset. Moreover, real-life traders must take into

_Mathematics_ **2022**, _10_, 3302 11 of 13

consideration the risks involved in algorithmic trading and their associated costs before
they are engaged in such a venture. Unlike past papers, the current paper utilized a
holistic methodology towards trading taking into consideration human behavior and sentiments that were collected from social media that can add insights about future price
movements of financial assets that cannot be revealed from past raw data derived from the
financial markets. Moreover, we provided practical suggestions for manual traders and
algorithmic traders.

The future of algorithmic trading systems is exciting. We estimate that new techniques will be developed using an even larger amount of data sets that would be analyzed
simultaneously using quantum computers producing faster and more accurate price predictions. New devices such as brain scanners and eye trackers will provide the systems
with a full human emotion profile that will derive prediction accuracy even further. Deepening algorithmic trading system entrenchment in the finance sector might enhance the
human–computer conflict. Ref. [61], for example, showed that people perceive AI as more
capable of impartiality than humans. The human–machine interface should be carefully
balanced in the future to ensure that the advantages of both machines and humans can be
incorporated successfully.

**Funding:** This research was funded by Western Galilee College.

**Data Availability Statement:** Not Applicable.

**Conflicts of Interest:** The authors declare no conflict of interest.

**References**

1. Boming, H.; Yuxiang, H.; Xu, L.D.; Zheng, L.; Zhou, Z. Automated trading systems statistical and machine learning methods and
hardware implementation: A survey. _Enterprize Inf._ _Syst._ **2019**, _13_, 132–144.
2. McGroarty, F.; Booth, A.; Gerding, E.; Chinthalapati, V.L.R. High frequency trading strategies, market fragility and price spikes:
An agent based model perspective. _Ann._ _Oper._ _Res._ **2018**, _282_, 217–244. [[CrossRef]](http://doi.org/10.1007/s10479-018-3019-4)
3. Goldstein, M.A.; Kwan, A.; Richard, P. High Frequency Trading Strategies. 2021. Available online: [https://papers.ssrn.com/sol3](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2973019)
[/papers.cfm?abstract_id=2973019 (accessed on 12 June 2022).](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2973019)
4. Rundo, F.; Trenta, F.; di Stallo, A.L.; Battiato, S. Grid Trading System Robot (GTSbot): A Novel Mathematical Algorithm for
Trading FX Market. _Appl._ _Sci._ **2019**, _9_, 1796. [[CrossRef]](http://doi.org/10.3390/app9091796)
5. Lahmiri, S.; Bekiros, S. Intelligent forecasting with machine learning trading systems in chaotic intraday Bitcoin market. _Chaos,_
_Solitons Fractals_ **2020**, _133_, 109641. [[CrossRef]](http://doi.org/10.1016/j.chaos.2020.109641)
6. Bao, T.; Nekrasova, E.; Neugebauer, T.; Riyanto, Y.E. Algorithmic Trading in Experimental Markets with Human Traders: A
Literature Survey. In _Handbook_ _of_ _Experimental_ _Finance_ ; Sascha, F., Ernan, H., Eds.; Edward Elgar Publishing: Cheltenham,
UK, 2021.
7. Dubey, R.K. Algorithmic Trading: The Intelligent Trading Systems and Its Impact on Trade Size. _Expert_ _Syst._ _Appl._ **2022**,
_202_, 117279. [[CrossRef]](http://doi.org/10.1016/j.eswa.2022.117279)
8. Ashta, A.; Herrmann, H. Artificial Intelligence and fintech: An overview of opportunities for banking, investments and
microfinance. _Strateg._ _Chang._ **2021**, _30_, 211–222. [[CrossRef]](http://doi.org/10.1002/jsc.2404)
9. Kahneman, D.; Tversky, A. Prospect Theory: An Analysis of Decision Under Risk. _Econometrica_ **1977**, _47_, 263–291. [[CrossRef]](http://doi.org/10.2307/1914185)
10. Cohen, G.; Qadan, M. The Complexity of Cryptocurrencies Algorithmic Trading. _Mathematics_ **2022**, _10_, 2037. [[CrossRef]](http://doi.org/10.3390/math10122037)
11. Fischer, T.; Krauss, C. Deep learning with long short-term memory networks for financial market predictions. _Eur._ _J. Oper._ _Res._
**2018**, _270_, 654–669. [[CrossRef]](http://doi.org/10.1016/j.ejor.2017.11.054)
12. Lu, T.-H.; Shiu, Y.-M. Pinpoint and synergistic trading strategies of candlesticks. _Int._ _J. Econ._ _Finance_ **2011**, _3_, 234–244. [[CrossRef]](http://doi.org/10.5539/ijef.v3n1p234)
13. Jay, P.; Kalariya, V.; Parmar, P.; Tanwar, S.; Kumar, N.; Alazab, M. Stochastic Neural Networks for Cryptocurrency Price Prediction.
_IEEE Access_ **2020**, _8_, 82804–82818. [[CrossRef]](http://doi.org/10.1109/ACCESS.2020.2990659)
14. Senthuran, G.; Halgamuge, M. Prediction of Cryptocurrency Market Price Using Deep Learning and Blockchain Information:
Bitcoin and Ethereum. In _Essentials of Blockchain Technology_ ; Taylor & Francis: New York, NY, USA, 2019; pp. 349–364.
15. Vapnik, V.N. The Nature of Statistical Learning Theory. _Technometrics_ **1997**, _38_, 409.
16. Xiao, C.; Xia, W.; Jiang, J. Stock price forecast based on combined model of ARI-MA-LS-SVM. _Neural Comput._ _Appl._ **2020**, _32_,
5379–5388. [[CrossRef]](http://doi.org/10.1007/s00521-019-04698-5)
17. Huang, W.; Nakamori, Y.; Wang, S.-Y. Forecasting stock market movement direction with support vector machine. _Comput._ _Oper._
_Res._ **2005**, _32_, 2513–2522. [[CrossRef]](http://doi.org/10.1016/j.cor.2004.03.016)
18. Thu, N.T.; Xuan, V.D. Using support vector machine in Forex predicting. In Proceedings of the IEEE International Conference on
Innovative Research and Development, Khlong Nueng, Thailand, 11–12 May 2018; pp. 1–5.

_Mathematics_ **2022**, _10_, 3302 12 of 13

19. Akyildirim, E.; Goncu, A.; Sensoy, A. Prediction of cryptocurrency returns using machine learning. _Ann._ _Oper._ _Res._ **2020**,
_297_, 3–36. [[CrossRef]](http://doi.org/10.1007/s10479-020-03575-y)
20. Hitam, N.A.; Ismail, A.R.; Samsudin, R.; Alkhammash, E.H. The Effect of Kernel Functions on Cryptocurrency Prediction Using
Support Vector Machines. In _Advances on Intelligent Informatics and Computing_ ; Saeed, F., Mohammed, F., Ghaleb, F., Eds.; Springer:
Berlin/Heidelberg, Germany, 2022.
21. Gers, F.; Schmidhuber, E. LSTM recurrent networks learn simple context-free and context-sensitive languages. _IEEE Trans._ _Neural_
_Netw._ **2001**, _12_, 1333–1340. [[CrossRef] [PubMed]](http://doi.org/10.1109/72.963769)
22. Gers, F.; Schraudolph, N.; Schmidhuber, J. Learning precise timing with LSTM recurrent networks. _J. Mach._ _Learn._ _Res._ **2002**,
_3_, 115–143.
23. Borovkova, S.; Tsiamas, I. An ensemble of LSTM neural networks for high-frequency stock market classification. _J. Forecast._ **2019**,
_38_, 600–619. [[CrossRef]](http://doi.org/10.1002/for.2585)
24. Nelson, D.M.Q.; Pereira, A.C.; de Oliveira, O. Stock market’s price movement prediction with LSTM neural networks. In
Proceedings of the International Joint Conference on Neural Networks (IJCNN), Anchorage, AK, USA, 14–19 May 2017;
pp. 1419–1426.
25. Ji, S.; Kim, J.; Im, H. A Comparative Study of Bitcoin Price Prediction Using Deep Learning. _Mathematics_ **2019**, _7_, 898. [[CrossRef]](http://doi.org/10.3390/math7100898)
26. Alessandretti, L.; ElBahrawy, A.; Aiello, L.M.; Baronchelli, A. Anticipating cryptocurrency prices using machine learning.
_Complexity_ **2018**, _2018_, 8983590. [[CrossRef]](http://doi.org/10.1155/2018/8983590)
27. Banik, S.; Sharma, N.; Mangla, M.; Mohanty, S.N.; Shitharth, S. LSTM based decision support system for swing trading in stock
market. _Knowledge-Based Syst._ **2021**, _239_, 107994. [[CrossRef]](http://doi.org/10.1016/j.knosys.2021.107994)
28. Ghosh, P.; Neufeld, A.; Sahoo, J.K. Forecasting directional movements of stock prices for intraday trading using LSTM and
random forests. _Finance Res._ _Lett._ **2022**, _46_, 102280. [[CrossRef]](http://doi.org/10.1016/j.frl.2021.102280)
29. Wei, D. Prediction of Stock Price Based on LSTM Neural Network. In Proceedings of the 2019 International Conference on
Artificial Intelligence and Advanced Manufacturing (AIAM), Dublin, Ireland, 17–19 October 2019; pp. 544–547.
30. Chourmouziadis, K.; Chourmouziadou, D.K.; Chatzoglou, P.D. Embedding Four Medium-Term Technical Indicators to an
Intelligent Stock Trading Fuzzy System for Predicting: A Portfolio Management Approach. _Comput._ _Econ._ **2020**, _57_, 1183–1216.

[[CrossRef]](http://doi.org/10.1007/s10614-020-10016-2)
31. Hadavandi, E.; Shavandi, H.; Ghanbari, A. Integration of genetic fuzzy systems and artificial neural networks for stock price
forecasting. _Knowledge-Based Syst._ **2010**, _23_, 800–808. [[CrossRef]](http://doi.org/10.1016/j.knosys.2010.05.004)
32. Zarandi, M.H.F.; Hadavandi, E.; Turksen, I.B. A hybrid fuzzy intelligent agent-based system for stock price prediction. _Int._ _J._
_Intell._ _Syst._ **2012**, _27_, 947–969. [[CrossRef]](http://doi.org/10.1002/int.21554)
33. Satyaloka, D.; Giamiko, S.; Hidayat, A. SK-MOEFS Multi-Objective Evolutionary Fuzzy System Library effectiveness as UserFriendly Cryptocurrency Prediction Tool. In Proceedings of the 2021 International Conference on Artificial Intelligence and Big
Data Analytics, Bandung, Indonesia, 27–29 October 2021; pp. 1–5. [[CrossRef]](http://doi.org/10.1109/icaibda53487.2021.9689763)
34. Chandar, S.K. Stock market prediction using subtractive clustering for a neuro fuzzy hybrid approach. _Clust._ _Comput._ **2017**, _22_,
13159–13166. [[CrossRef]](http://doi.org/10.1007/s10586-017-1321-6)
35. Kleinbrod, V.; Xiaoming, L. Order Flow, Volatility and Fuzzy Logic: Technical Analyses for Currency Trading. 2017. Available
online: [https://ssrn.com/abstract=2976558 (accessed on 10 April 2022).](https://ssrn.com/abstract=2976558)
36. Maciel, L.; Ballini, R.; Gomide, F. Evolving Possibilistic Fuzzy Modeling for Realized Volatility Forecasting with Jumps. _IEEE_
_Trans._ _Fuzzy Syst._ **2016**, _25_, 302–314. [[CrossRef]](http://doi.org/10.1109/TFUZZ.2016.2578338)
37. Atsalakis, G.S.; Atsalaki, I.G.; Pasiouras, F.; Zopounidis, C. Bitcoin price forecasting with neuro-fuzzy techniques. _Eur._ _J. Oper._
_Res._ **2019**, _276_, 770–780. [[CrossRef]](http://doi.org/10.1016/j.ejor.2019.01.040)
38. Othman, S.; Schneider, E. Decision making using fuzzy logic for stock trading. In Proceedings of the 2010 International Symposium
on Information Technology, Kuala Lumpur, Malaysia, 15–17 June 2010; Volume 2, pp. 880–884. [[CrossRef]](http://doi.org/10.1109/itsim.2010.5561564)
39. Zhou, X.S.; Dong, M. Can Fuzzy Logic Make Technical Analysis 20/20? _Financ._ _Anal._ _J._ **2004**, _60_, 54–75. [[CrossRef]](http://doi.org/10.2469/faj.v60.n4.2637)
40. Ahmad, M.; Soeparno, H.; Napitupulu, T.A. Stock Trading Alert: With fuzzy knowledge-based systems and technical analysis. In
Proceedings of the 2020 International Conference on Information Technology Systems and Innovation (ICITSI), Bandung-Padang,
Indonesia, 19–23 October 2020; pp. 155–160.
41. Cohen, G. Intraday Trading of Precious Metals Futures Using Algorithmic Systems. _Chaos Solitons Fractals_ **2022**, _154_, 111676.
42. Choijil, E.; Méndez, C.E.; Wong, W.-K.; Vieito, J.P.; Batmunkh, M.-U. Thirty years of herd behavior in financial markets:
A bibliometric analysis. _Res._ _Int._ _Bus._ _Finance_ **2021**, _59_, 101506. [[CrossRef]](http://doi.org/10.1016/j.ribaf.2021.101506)
43. Vo, X.V.; Phan, D.B.A. Herd behavior and idiosyncratic volatility in a frontier market. _Pac.-Basin Finance J._ **2018**, _53_, 321–330.

[[CrossRef]](http://doi.org/10.1016/j.pacfin.2018.10.005)
44. Tan, S.D.; Tas, O. Social Media Sentiment in International Stock Returns and Trading Activity. _J. Behav._ _Finance_ **2020**, _22_, 221–234.

[[CrossRef]](http://doi.org/10.1080/15427560.2020.1772261)
45. Abraham, J.; Higdon, D.; Nelson, J.; Ibarra, J. Cryptocurrency price prediction using tweet volumes and sentiment analysis. _SMU_
_Data Sci._ _Rev._ **2018**, _1_, 1.
46. Liu, Y.; Yang, A.; Zhang, J.; Yao, J. An Optimal Stopping Problem of Detecting Entry Points for Trading Modeled by Geometric
Brownian Motion. _Comput._ _Econ._ **2019**, _55_, 827–843. [[CrossRef]](http://doi.org/10.1007/s10614-019-09915-w)

_Mathematics_ **2022**, _10_, 3302 13 of 13

47. Bizzi, L.; Labban, A. The double-edged impact of social media on online trading: Opportunities, threats, and recommendations
for organizations. _Bus._ _Horizons_ **2019**, _62_, 509–519. [[CrossRef]](http://doi.org/10.1016/j.bushor.2019.03.003)
48. Picasso, A.; Merello, S.; Ma, Y.; Oneto, L.; Cambria, E. Technical analysis and sentiment embeddings for market trend prediction.
_Expert Syst._ _Appl._ **2019**, _135_, 60–70. [[CrossRef]](http://doi.org/10.1016/j.eswa.2019.06.014)
49. Wołk, K. Advanced social media sentiment analysis for short-term cryptocurrency price prediction. _Expert Syst._ **2020**, _37_, 2.

[[CrossRef]](http://doi.org/10.1111/exsy.12493)
50. Sohangir, S.; Wang, D.; Pomeranets, A.; Khoshgoftaar, T.M. Big Data: Deep Learning for financial sentiment analysis. _J. Big Data_
**2018**, _5_, 3. [[CrossRef]](http://doi.org/10.1186/s40537-017-0111-6)
51. Kim, Y.B.; Lee, J.; Park, N.; Choo, J.; Kim, J.H.; Kim, C.H. When bitcoin encounters information in an online forum: Using text
mining to analyze user opinions and predict value fluctuation. _PLoS ONE_ **2017**, _12_, e0177630. [[CrossRef]](http://doi.org/10.1371/journal.pone.0177630)
52. Valencia, F.; Gómez-Espinosa, A.; Valdés-Aguirre, B. Price Movement Prediction of Cryptocurrencies Using Sentiment Analysis
and Machine Learning. _Entropy_ **2019**, _21_, 589. [[CrossRef]](http://doi.org/10.3390/e21060589)
53. Ren, R.; Wu, D.D.; Liu, T. Forecasting Stock Market Movement Direction Using Sentiment Analysis and Support Vector Machine.
_IEEE Syst._ _J._ **2018**, _13_, 760–770. [[CrossRef]](http://doi.org/10.1109/JSYST.2018.2794462)
54. Huang, J.; Liu, J. Using social media mining technology to improve stock price forecast accuracy. _J. Forecast._ **2019**, _39_, 104–116.

[[CrossRef]](http://doi.org/10.1002/for.2616)
55. Heinz, A.; Jamaloodeen, M.; Saxena, A.; Pollacia, L. Bullish and Bearish Engulfing Japanese Candlestick patterns: A statistical
analysis on the S&P 500 index. _Q. Rev._ _Econ._ _Finance_ **2020**, _79_, 221–244. [[CrossRef]](http://doi.org/10.1016/j.qref.2020.06.006)
56. Tsung, H.L.; Yung, M.S.; Tsung, C.L. Profitable candlesticks trading strategies-the evidence from a new perspective. _Rev._ _Finance_
_Econ._ **2012**, _21_, 63–68.
57. Cohen, G. Optimizing Candlesticks Patterns for Bitcoin’s Trading Systems. _Rev._ _Quant._ _Finance Account._ **2021**, _57_, 1155–1167.

[[CrossRef]](http://doi.org/10.1007/s11156-021-00973-6)
58. Cohen, G. Algorithmic Strategies for Precious Metals Price Forecasting. _Mathematics_ **2022**, _10_, 1134. [[CrossRef]](http://doi.org/10.3390/math10071134)
59. Lauguico, S.; Ii, R.C.; Alejandrino, J.; Macasaet, D.; Tobias, R.R.; Bandala, A.; Dadios, E. A Fuzzy Logic-Based Stock Market Trading
Algorithm Using Bollinger Bands. In Proceedings of the 2019 IEEE 11th International Conference on Humanoid, Nanotechnology, Information Technology, Communication and Control, Environment, and Management (HNICEM), Laoag, Philippines,
29 November–1 December 2019; pp. 1–6. [[CrossRef]](http://doi.org/10.1109/hnicem48295.2019.9072734)
60. Ni, Y.; Day, M.-Y.; Huang, P.; Yu, S.-R. The profitability of Bollinger Bands: Evidence from the constituent stocks of Taiwan 50.
_Phys._ _A Stat._ _Mech._ _Its Appl._ **2020**, _551_, 124144. [[CrossRef]](http://doi.org/10.1016/j.physa.2020.124144)
61. Claudy, M.C.; Aquino, K.; Graso, M. Artificial Intelligence Can’t be Charmed: The Effect of Impartially on Laypeople’s Algorithmic
Preferences. _Front._ _Psychol._ **2022**, _13_, 898027. [[CrossRef]](http://doi.org/10.3389/fpsyg.2022.898027)

