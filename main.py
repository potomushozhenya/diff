import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.use('Qt5Agg')
#Исп. условия порядка для двухэтапного ЯМРК второго порядка
#построить расчетную схему второго порядка при значении параметра c2 в указанном варианте
#c2 = 0.4 A=-3 B=2.5 C=1 оппонент-метод y(x0 + h) = y0 + h/2(f(x0,y0) + f(x0+h,y0 + hf(x0,y0))
#
c2 = 0.4
A = -3
B = 2.5
C = 1
call = 0
atol = 10**-12
rtol = 10**-6
hMax = 0.1

def y(x):
    return [np.exp(np.sin(x**2)), np.exp(B*np.sin(x**2)), C*np.sin(x**2)+A, np.cos(x**2)]


def f(x, y):
    global call
    call += 1
    if y[1] < 0 or y[0] < 0:
        print("Wrong argument")
        return np.full(shape=4, fill_value=np.nan)
    return [2*x*(y[1]**(1/B))*y[3], 2*B*x*np.exp((B/C)*(y[2]-A))*y[3], 2*C*x*y[3], -2*x*np.log(y[0])]


def multiply(x, y):
    res = []
    for i in range(len(y)):
        res.append(x * y[i])
    return res


def sumVec(x, y):
    s = len(x)
    res = []
    if s != len(y):
        print("Error!")
    else:
        for i in range(s):
            res.append(x[i] + y[i])
    return res


def norm(x):
    res = 0
    for i in range(len(x)):
        res += x[i]**2
    return res**0.5


def stepRungeKutta(x0, y0, h):
    k1 = f(x0, y0)
    if np.isnan(k1[0]):
        return np.full(shape=4, fill_value=np.nan)
    k2 = f(x0 + c2 * h, sumVec(y0, multiply(c2, multiply(h, k1))))
    return sumVec(y0, multiply(h, sumVec(multiply((1 - 1 / (2 * c2)), k1), multiply((1 / (2 * c2)), k2))))


def stepEuler(x0, y0, h):
    return sum(y0, multiply(h,f(x0, y0)))


def stepHeun(x0, y0, h):
    k1 = f(x0, y0)
    k2 = f(x0 + h, sumVec(y0, k1))
    if np.isnan(k1[0]) or np.isnan(k2[0]):
        return np.full(shape=4, fill_value=np.nan)
    return sumVec(y0, multiply(h, sumVec(multiply(0.5, k1), multiply(0.5, k2))))


def stepRungeKuttaThird(x0, y0, h):
    k1 = f(x0, y0)
    k2 = f(x0 + h/2, sumVec(y0, multiply(h/2, k1)))
    k3 = f(x0 + h, sumVec(sumVec(y0, multiply(-h, k1)), multiply(2*h, k2)))
    if np.isnan(k1[0]) or np.isnan(k2[0]):
        return np.full(shape=4, fill_value=np.nan)
    return sumVec(y0, multiply(h, sumVec(multiply(1/6, k1), sumVec(multiply(4/6, k2), multiply(1/6, k3)))))


def ruleRunge(main, sub, p):
    return multiply(2**p - 1, sumVec(sub, multiply(-1, main)))


def hTol(h, tol, R, p):
    return ((tol*norm(R))**(1/p)) * h/2


def fixedStepSub(x0, xFin, y0, hMain):
    hSub = hMain/2
    i = 0
    yr = 0
    y0RKMain = y0
    y0RKSub = y0
    y0HMain = y0
    y0HSub = y0
    xStep = []
    RstepRK = []
    RstepH = []
    yStepRK = []
    yStepH = []
    while i < (xFin / hSub):
        if i % 2 != 0:
            xStep.append(x0)
            y0RKMain = stepRungeKutta(x0, y0RKMain, hMain)
            y0RKSub = stepRungeKutta(x0, y0RKSub, hSub)
            y0HMain = stepHeun(x0, y0HMain, hMain)
            y0HSub = stepHeun(x0, y0HSub, hSub)
            x0 += hSub
            yStepRK.append(y0RKMain)
            yStepH.append(y0HMain)
            RstepRK.append(ruleRunge(y0RKMain, y0RKSub, 2))
            RstepH.append(ruleRunge(y0HMain, y0HSub, 2))
            if (np.isnan(y0RKMain[0]) or np.isnan(y0RKMain[1]) or np.isnan(y0RKMain[2]) or np.isnan(y0RKMain[3])) and (np.isnan(y0HMain[0]) or np.isnan(y0HMain[1]) or np.isnan(y0HMain[2]) or np.isnan(y0HMain[3])):
                break
            #yr = y(x0)
            #print(sumVec(yr, multiply(-1, y0)))
        else:
            y0RKSub = stepRungeKutta(x0, y0RKSub, hSub)
            y0HSub = stepHeun(x0, y0HSub, hSub)
            x0 += hSub
        i += 1
    return xStep, yStepRK, yStepH, RstepH, RstepRK


def fixedStep(x0Init, xFin, y0Init):
    yRK = []
    yH = []
    x0 = x0Init
    y0 = y0Init
    x = []
    RRK = []
    RH = []
    hAll = []
    for k in range(7):
        h = 1 / (2**k)
        hAll.append(h)
        xStep, yStepRK, yStepH, RstepH, RstepRK = fixedStepSub(x0, xFin, y0, h)
        x.append(xStep)
        yRK.append(yStepRK)
        yH.append(yStepH)
        RH.append(RstepH)
        RRK.append(RstepRK)
        y0 = y0Init
        x0 = x0Init
    return x, yRK, yH, RH, RRK, hAll


def optimalSub(x0, xFin, y0, R, hInit):
    h = hTol(hInit, 1e-5, R, 2)
    return fixedStepSub(x0, xFin, y0, h)


def optimalStep(x0, xFin, y0, R, h):
    return optimalSub(x0, xFin, y0, R, h)


def printYPlots(x, y, z=None):
    for i in range(len(x)):
        fig, axs = plt.subplots(2)
        axs[0].plot(x[i], y[i])
        if z!=None:
            currZ = [norm(z[i][j]) for j in range(len(z[i]))]
            axs[1].plot(x[i], currZ)
    plt.show()


def yGenerate(hList):
    ylist = []
    for h in hList:
        x = 0
        yStep = []
        while x < 5:
            yStep.append(y(x))
            x+=h
        ylist.append(yStep)
    return ylist


def plotDiff(x, yRK, y):
    for i in range(len(x)):
        fig, axs = plt.subplots(2)
        axs[0].plot(np.log10(x[i]), np.log10(yRK[i]))
        axs[0].plot(np.log(x[i], 2*np.log(x[i])), linestyle='dashed')
        axs[1].plot(x[i], y[i])
    plt.show()


def printAuto(x, y, z=None):
    currY = []
    if z!=None:
        currY = [np.log10(norm(z[i]-y[i])) for i in range(len(y))]
    else:
        currY = y
    plt.plot(x, currY)
    plt.show()


def printAutoStep(x, y):
    hTakeX = [x[i][0] for i in range(len(x))]
    hTakeY = [x[i][1] for i in range(len(x))]
    plt.plot(hTakeX, hTakeY)
    hFakeX = [y[i][0] for i in range(len(y))]
    hFakeY = [y[i][1] for i in range(len(y))]
    plt.plot(hFakeX, hFakeY, marker='*', linewidth=0)
    plt.show()


def rtolGraph(isOpponent=False):
    global call, rtol
    call = 0
    callList =[]
    rtolList = [1e-4, 1e-5, 1e-6, 1e-7, 1e-8]
    for el in rtolList:
        rtol = el
        autoStep(0, 5.0, y(0), 10 ** -6, isOpponent)
        callList.append(call)
        call=0
    plt.plot(np.log10(rtolList), np.log10(callList))
    plt.show()


def printAutoStepDiff(x, y, yReal):
    plt.plot(x, y)
    plt.plot(x, yReal, linestyle='dashed')
    plt.show()


def autoStep(x0, xFin, y0, tol, isOpponent=False):
    p = 2
    if isOpponent:
        func = stepHeun
    else:
        func = stepRungeKutta
    f0 = f(x0, y0)
    delta0 = (1 / max(abs(x0), abs(xFin)))**(p + 1) + norm(f0)**(p+1)
    h0 = (tol / delta0) ** (1 / (p + 1))
    u = sumVec(y0, multiply(h0, f0))
    f1 = f(x0 + h0, u)
    delta1 = (1 / max(abs(x0+h0), abs(xFin)))**(p + 1) + norm(f1)**(p + 1)
    h1 = (tol / delta1) ** (1 / (p + 1))
    h, points_x, points_y = min(h0, h1), [x0], [y0]
    take_h, fake_h = [], []
    flag = True
    y_h, y_2h = [], []
    while points_x[-1] + h < xFin:
        if flag:
            y_h = func(points_x[-1], points_y[-1], h)
        else:
            y_h = np.copy(y_2h)
        flag = True
        y_2h_0 = func(points_x[-1], points_y[-1], h / 2)
        y_2h = func(points_x[-1] + h / 2, y_2h_0, h / 2)
        r = norm(multiply(1 / (1 - 2 ** (-p)), sumVec(y_2h, multiply(-1, y_h))))
        sigma = rtol * np.linalg.norm(y_h) + atol
        if r > sigma * 2 ** p:
            fake_h.append((points_x[-1], h))
            h /= 2
            flag = False
        elif r > sigma:
            take_h.append((points_x[-1], h))
            points_y.append(y_2h)
            points_x.append(points_x[-1] + h)
            h /= 2
        elif r > sigma / (2 ** (p + 1)):
            take_h.append((points_x[-1], h))
            points_y.append(y_h)
            points_x.append(points_x[-1] + h)
        else:
            take_h.append((points_x[-1], h))
            points_y.append(y_h)
            points_x.append(points_x[-1] + h)
            h = min(2*h, hMax)
    h = xFin - points_x[-1]
    if h > 1e-6:
        points_y.append(func(points_x[-1], points_y[-1], h))
        points_x.append(points_x[-1] + h)
    return np.array(points_x), np.array(points_y), np.array(take_h), np.array(fake_h)


x, yRK, yH, RH, RRK, h = fixedStep(0, 5.0, y(0))
xOpt, yRKOpt, yHOpt, RHOpt, RRKOpt = optimalStep(0, 5.0, y(0), RRK[-1][-1], h[6])
xAuto, yAuto, hTake, hFake = autoStep(0, 5.0, y(0), 10**-4)
printYPlots(x, yRK, RRK) #РК
printYPlots(x, yH, RH) #Хойн
printAuto(xOpt, yRKOpt) #Оптимальный
yAutoReal = [y(el) for el in xAuto]
printAuto(xAuto, yAuto, yAutoReal) #Автоматический
printAutoStep(hTake, hFake) #Шаги
rtolGraph()

#printAutoStepDiff(xAuto, yAuto, yAutoReal)

#x, yRK, yH, RH, RRK = fixedStepSub(0, 5.0, y(0), 10**-4, False)
#y = yGenerate(h)
#plotDiff(x, yRK, y)
