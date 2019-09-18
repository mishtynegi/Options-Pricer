# -*- coding: utf-8 -*-
import bottle
from bottle import route, run, template, request, response, static_file
import base64
import os
from black_scholes import BlackScholes
from implied_volatility import NewtonRaphson
from asian_options import AsianOption
from AmericanOption import AmericanOption_BiTree
from BasketOption_TwoAssets import BasketOption
from Extension_MC import ExtensionMonteCarlo

bottle.BaseRequest.MEMFILE_MAX = 1024 * 1024 # (or whatever you want)
app = bottle.app()

@app.hook('after_request')
def enable_cors():
        # set CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, OPTIONS'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    response.headers['Access-Control-Allow-Headers'] = 'Access-Control-*, Origin, Accept, Content-Type, X-Requested-With, X-CSRF-Token, Cache-Control, X-File-Name, If-Modified-Since, User-Agent, Depth, X-File-Size'
    response.headers['Access-Control-Expose-Headers'] = 'Access-Control-*'
    # print(response)

@app.route('/static/<filename:path>', method=['OPTIONS', 'GET'])
def send_static(filename):
    return static_file(filename, root='data')

@app.route('/home/<filename:path>')
def index(filename):
    return static_file(filename, root='views')

@app.route('/blackScholes', method=['OPTIONS', 'POST'])
def blackScholes():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    repoRate = float(data['repoRate'])/100
    optionType = data['optionType']
    time = float(data['time'])

    blackScholes = BlackScholes(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, q=repoRate)

    if(optionType == 'call'):
        return str(round(blackScholes.call(),2))
    else:
        return str(round(blackScholes.put(),2))

@app.route('/europeanOptionMC', method=['OPTIONS', 'POST'])
def europeanOptionMC():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    numOfSteps = int(data['numOfSteps'])
    optionType = data['optionType']
    time = float(data['time'])

    extensionMC = ExtensionMonteCarlo(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, n=numOfSteps, option_type=optionType)

    return str(round(extensionMC.european_payoff(), 2))

@app.route('/impliedVolatility', method=['OPTIONS', 'POST'])
def impliedVolatility():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    premium = float(data['premium'])
    interestRate = float(data['interestRate'])/100
    repoRate = float(data['repoRate'])/100
    optionType = data['optionType']
    time = float(data['time'])

    impliedVolatility = NewtonRaphson(S=stockPrice, K=strike, r=interestRate, T=time, q=repoRate, type_=optionType)

    return str(round(NewtonRaphson.calc_σ(impliedVolatility, premium), 4))

@app.route('/asianOptionCF', method=['OPTIONS', 'POST'])
def asianOptionCF():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    numOfObs = int(data['numOfObs'])
    optionType = data['optionType']
    time = float(data['time'])

    asianOption = AsianOption(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, n=numOfObs, option_type=optionType)

    return str(round(asianOption.geo_std_MC(), 3))

@app.route('/asianOptionMC', method=['OPTIONS', 'POST'])
def asianOptionMC():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    numOfObs = int(data['numOfObs'])
    optionType = data['optionType']
    time = float(data['time'])
    numOfPaths = int(data['numOfPaths'])
    controlVariate = data['controlVariate']

    asianOption = AsianOption(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, n=numOfObs, option_type=optionType, m=numOfPaths)

    if(controlVariate == 'yes'):
        Zmean, confCV = asianOption.control_variate()
        return str(round(Zmean, 4)), str([round(x,4) for x in confCV])
    else:
        return str(round(asianOption.arith_std_MC(), 4))

@app.route('/americanOptionBN', method=['OPTIONS', 'POST', 'GET'])
def americanOptionBN():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    numOfSteps = int(data['numOfSteps'])
    optionType = data['optionType']
    time = float(data['time'])

    americanOption = AmericanOption_BiTree(S=stockPrice, K=strike, r=interestRate, T=time, sigma=volatility, step_num=numOfSteps, option_type=optionType)

    return str(round(americanOption.get_value(), 2))


@app.route('/americanOptionMC', method=['OPTIONS', 'POST', 'GET'])
def americanOptionMC():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    numOfSteps = int(data['numOfSteps'])
    numOfPaths = int(data['numOfPaths'])
    optionType = data['optionType']
    time = float(data['time'])

    extensionMC = ExtensionMonteCarlo(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, n=numOfSteps, m=numOfPaths, option_type=optionType)

    return str(round(extensionMC.american_payoff(), 2))

@app.route('/basketOptionCF', method=['OPTIONS', 'POST', 'GET'])
def basketOptionCF():
    data = request.json

    print(data)
    strike = float(data['strike'])
    stockPrice1 = float(data['stockPrice1'])
    stockPrice2 = float(data['stockPrice2'])
    volatility1 = float(data['volatility1'])/100
    volatility2 = float(data['volatility2'])/100
    interestRate = float(data['interestRate'])/100
    coefficient = float(data['coefficient'])
    optionType = data['optionType']
    time = float(data['time'])

    basketOption = BasketOption(S=[stockPrice1, stockPrice2], K=strike, r=interestRate, T=time, σ=[volatility1, volatility2], ρ=[[1, coefficient], [coefficient,1]], option_type=optionType)

    return str(round(basketOption.closed_form(), 2))

@app.route('/basketOptionMC', method=['OPTIONS', 'POST', 'GET'])
def basketOptionMC():
    data = request.json

    strike = float(data['strike'])
    stockPrice1 = float(data['stockPrice1'])
    stockPrice2 = float(data['stockPrice2'])
    volatility1 = float(data['volatility1'])/100
    volatility2 = float(data['volatility2'])/100
    interestRate = float(data['interestRate'])/100
    numOfObs = int(data['numOfObs'])
    coefficient = float(data['coefficient'])
    optionType = data['optionType']
    controlVariate = data['controlVariate']
    time = float(data['time'])

    basketOption = BasketOption(S=[stockPrice1, stockPrice2], K=strike, r=interestRate, T=time, σ=[volatility1, volatility2], ρ=[[1, coefficient], [coefficient,1]], n=numOfObs, option_type=optionType)

    if(controlVariate == 'yes'):
        Zmean, confCV = basketOption.control_variate()
        return str(round(Zmean, 4)), str([round(x,4) for x in confCV])
    else:
        return str(round(basketOption.arith_std_MC(), 2))

@app.route('/barrierOption', method=['OPTIONS', 'POST'])
def barrierOption():
    data = request.json

    strike = float(data['strike'])
    stockPrice = float(data['stockPrice'])
    volatility = float(data['volatility'])/100
    interestRate = float(data['interestRate'])/100
    barrier = float(data['barrier'])
    optionType = data['optionType']
    barrierType = data['barrierType']
    mainBarrierType = data['mainBarrierType']
    numOfSteps = int(data['numOfSteps'])
    numOfPaths = int(data['numOfPaths'])
    time = float(data['time'])

    # print(data)
    extensionMC = ExtensionMonteCarlo(S=stockPrice, K=strike, r=interestRate, T=time, σ=volatility, n=numOfSteps, m=numOfPaths, option_type=optionType)

    if(mainBarrierType == "in"):
        if(barrierType == "up"):
            return str(round(extensionMC.barrier_upandin(barrier), 2))
        else:
            return str(round(extensionMC.barrier_downandin(barrier), 2))
    else:
        if(barrierType == "up"):
            return str(round(extensionMC.barrier_upandout(barrier), 2))
        else:
            return str(round(extensionMC.barrier_downandout(barrier), 2))

app.run(host='localhost', port=8080)
