# utils.py
# includes:
# - custmin <- custom minimizer
# - custcallnack <- custom callback for minimizer

from scipy.optimize import OptimizeResult


def custmin(
    fun,
    guess=20,
    maxfev=None,
    bigstep=None,
    smallstep=None,
    callback=None,
    tol=1,
):
    bestx = guess
    besty = fun(bestx)
    funcalls = 1
    niter = 0
    improved = True
    stop = False
    segment_check = True
    stepsize = bigstep
    tol = tol

    if callback is not None:
        callback(error=besty, x=bestx, nit=niter, nfev=funcalls, stepsize=stepsize)

    # main loop
    while improved and not stop and besty > tol:
        # Initialization, we are checking which direction we should go
        while segment_check and abs(stepsize) == bigstep and not stop and besty > tol:
            niter += 1
            improved = False
            for testx in [max(1, bestx - stepsize), max(1, bestx + stepsize)]:
                testy = fun(testx)
                funcalls += 1
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nit=niter,
                        nfev=funcalls,
                        stepsize=stepsize,
                    )

                if testy <= tol:
                    besty = testy
                    bestx = testx
                    stop = True
                    break

                if testy < besty:
                    if testx == max(1, bestx + stepsize):
                        stepsize = -stepsize
                    besty = testy
                    bestx = testx
                    improved = True
                    break

            if improved == False:
                stepsize = smallstep

            if besty <= tol:
                stop = True
                break

        if besty <= tol:
            stop = True
            break

        # we redo initialization for the small stepsize just to check direction
        if segment_check and stepsize == smallstep and not stop:
            niter += 1
            improved = False
            for testx in [max(bestx - stepsize, 1), max(bestx + stepsize, 1)]:
                testy = fun(testx)
                funcalls += 1
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nit=niter,
                        nfev=funcalls,
                        stepsize=stepsize,
                    )

                if testy <= tol:
                    besty = testy
                    bestx = testx
                    stop = True
                    break

                if testy < besty:
                    if testx < bestx:
                        stepsize = -stepsize
                    besty = testy
                    bestx = testx
                    improved = True
                    segment_check = False
                    break
                segment_check = False
                break

        # then we will continue in this direction
        if besty <= tol:
            stop = True
            break

        improved = False
        testx = bestx + stepsize
        testy = fun(testx)
        niter += 1
        funcalls += 1
        if testy <= tol:
            besty = testy
            bestx = testx
            stop = True
            break

        if testy < besty:
            besty = testy
            bestx = testx
            improved = True
            callback(error=besty, x=bestx, nit=niter, nfev=funcalls, stepsize=stepsize)

        if callback is not None and improved == False:
            callback(error=testy, x=testx, nit=niter, nfev=funcalls, stepsize=stepsize)

        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    return OptimizeResult(
        fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter >= 1)
    )


def custcallback(error=None, x=None, nit=None, nfev=None, stepsize=None):
    print(
        "iteration #{}:   x={}   error={}  function evaluated {} times step taken: {}".format(
            nit, x, error, nfev, stepsize
        )
    )