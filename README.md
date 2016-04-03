# GMM-demo
An interactive demonstration of fitting a Gaussian Mixture Model to data using expectation maximization.

A GMM is a probability density function of the form

![Probability Density Function](/docs/density.png)

This demonstration generates sample data from a GMM, makes a guess of the true distribution, and then iteratively improves that guess using the EM algorithm. The user interface allows you to control all of these parameters, to see the EM algorithm in action.


## Dependencies

- Python 2.7 or 3.3+
- Numpy
- Matplotlib
- Scipy


## Running

Run the script `GMM_demo.py` with your favorite method, e.g.

```bash
$ python GMM_demo.py
```

You will be greeted by this user interface.

![UI Demo](/docs/demo.png)

The sliders on the left control the underlying distribution, and the sliders on the right control the initial guess. If you change any of these parameters, you must click `Refit` to update the plot, as this part is computationally intensive.

The slider on the bottom controls the iteration of the EM algorithm being displayed. Iteration 0 is your initial guess. You do not need to click `Refit` after moving this slider, as this part is not computationally intensive.




## Author

Daniel Wysocki


## License

You are free to modify and redistribute this under the conditions of the MIT license.

Copyright 2016
