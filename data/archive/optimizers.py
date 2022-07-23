# utils.py
# includes:
# - custmin <- custom minimizer
# - custcallnack <- custom callback for minimizer

from scipy.optimize import OptimizeResult
import matplotlib.pyplot as plt
import ray


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
    fev_list = []
    fev_list.append((bestx, besty))
    funcalls = 1
    niter = 0
    improved = True
    stop = False
    segment_check = True
    stepsize = bigstep
    tol = tol

    if callback is not None:
        callback(
            error=besty,
            x=bestx,
            nit=niter,
            nfev=funcalls,
            stepsize=stepsize,
            fev_list=fev_list,
        )

    # main loop
    while improved and not stop and besty > tol:
        # Initialization, we are checking which direction we should go
        while (
            segment_check
            and abs(stepsize) == bigstep
            and not stop
            and besty > tol
        ):
            niter += 1
            improved = False
            for testx in [max(1, bestx - stepsize), max(1, bestx + stepsize)]:
                testy = fun(testx)
                fev_list.append((testx, testy))
                funcalls += 1
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nit=niter,
                        nfev=funcalls,
                        stepsize=stepsize,
                        fev_list=fev_list,
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
                fev_list.append((testx, testy))
                funcalls += 1
                if callback is not None:
                    callback(
                        error=testy,
                        x=testx,
                        nit=niter,
                        nfev=funcalls,
                        stepsize=stepsize,
                        fev_list=fev_list,
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
        fev_list.append((testx, testy))
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
            callback(
                error=besty,
                x=bestx,
                nit=niter,
                nfev=funcalls,
                stepsize=stepsize,
                fev_list=fev_list,
            )

        if callback is not None and improved == False:
            callback(
                error=testy,
                x=testx,
                nit=niter,
                nfev=funcalls,
                stepsize=stepsize,
                fev_list=fev_list,
            )

        if maxfev is not None and funcalls >= maxfev:
            stop = True
            break

    callback(
        error=testy,
        x=testx,
        nit=niter,
        nfev=funcalls,
        stepsize=stepsize,
        fev_list=fev_list,
    )

    return OptimizeResult(
        fun=besty, x=bestx, nit=niter, nfev=funcalls, success=(niter >= 1)
    )


def custcallback(
    error=None, x=None, nit=None, nfev=None, stepsize=None, fev_list=None
):
    print(
        "iteration #{}:   x={}   error={}  function evaluated {} times step taken: {}".format(
            nit, x, error, nfev, stepsize
        )
    )
    x_plot = [fev[0] for fev in fev_list]
    y_plot = [fev[1] for fev in fev_list]
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(x_plot, y_plot, "o", color="royalblue")
    ax.plot(x_plot, y_plot, "-", color="royalblue")

    # formatting
    ax.set_xlim(left=0)
    ax.set(ylabel="cost (log scale)")
    ax.set(xlabel="variable value")
    ax.set_xticks(x_plot)
    ax.set_yscale("log")

    plt.show()


def custmin_multicore(
    fun,
    guess=20,
    N_core=None,
    callback=None,
    tol=1,
):
    # adapt ray to function
    @ray.remote
    def f(x):
        return fun(x)

    # initialize list of results
    all_x = []
    all_y = []
    loop = 0

    # initialized finished as false
    finished = False

    # determine initial xmin and xmax
    xmin = int(max(1, guess - N_core / 2))
    xmax = int(guess + N_core / 2)

    # loop
    while finished == False:
        # build updated x_eval_list
        x_eval_list = [i for i in range(xmin, xmax, 1)]
        all_x.extend(x_eval_list)

        # evaluate all points in parallel
        futures = [f.remote(i) for i in x_eval_list]
        results = ray.get(futures)
        all_y.extend(results)

        # evaluate where is the min
        besty = min(results)
        index = results.index(besty)
        bestx = x_eval_list[index]

        # if tolerance is met
        if besty < tol:
            finished = True
            callback(
                all_x,
                all_y,
                bestx,
                besty,
                finished,
                loop,
            )
            break

        # if index is the min of x_eval_list
        if index == 0:
            callback(
                all_x,
                all_y,
                bestx,
                besty,
                finished,
                loop,
            )
            xmax = bestx + 1
            xmin = int(max(1, xmax - N_core))
            finished = False
            loop += 1

        # if index is the min of x_eval_list
        elif index == len(x_eval_list) - 1:
            callback(
                all_x,
                all_y,
                bestx,
                besty,
                finished,
                loop,
            )
            xmin = bestx
            xmax = xmin + N_core
            finished = False
            loop += 1

        # if we did not reach tol but the min is inside the segment
        # we finish
        else:
            finished = True
            callback(
                all_x,
                all_y,
                bestx,
                besty,
                finished,
                loop,
            )
            break

    return (
        bestx,
        besty,
        finished,
        all_x,
        all_y,
    )


def custcallback_multicore(
    all_x,
    all_y,
    bestx,
    besty,
    finished,
    loop,
):
    print(
        "loop #{}:   bestx={}   besty=error={} finished={}".format(
            loop,
            bestx,
            besty,
            finished,
        )
    )

    fig, ax = plt.subplots(figsize=(6, 2))
    ax.plot(all_x, all_y, "o", color="royalblue")

    # formatting
    ax.set_xlim(left=0)
    ax.set(ylabel="cost (log scale)")
    ax.set(xlabel="variable value")
    ax.set_xticks(all_x)
    ax.set_yscale("log")

    plt.show()
