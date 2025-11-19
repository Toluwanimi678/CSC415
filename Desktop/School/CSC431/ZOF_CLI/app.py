from flask import Flask, render_template, request
import math

app = Flask(__name__)


# ======================
# Utilities
# ======================

def parse_function(coeffs, const):
    def f(x):
        total = 0
        power = len(coeffs)
        for i, coef in enumerate(coeffs):
            total += coef * (x ** (power - i))
        total += const
        return total

    return f


def derivative(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)


# ======================
# Numerical Methods
# ======================

def bisection(f, a, b, tol, max_iter):
    iterations = []
    c_old = None

    for i in range(1, max_iter + 1):
        c = (a + b) / 2
        error = abs(c - c_old) if c_old is not None else None

        iterations.append([i, a, b, c, f(c), error])

        if f(c) == 0 or (error is not None and error < tol):
            return c, error, i, iterations

        if f(a) * f(c) < 0:
            b = c
        else:
            a = c

        c_old = c

    return c, error, max_iter, iterations


def regula_falsi(f, a, b, tol, max_iter):
    iterations = []
    x_old = None

    for i in range(1, max_iter + 1):
        x = (a * f(b) - b * f(a)) / (f(b) - f(a))
        error = abs(x - x_old) if x_old is not None else None

        iterations.append([i, a, b, x, f(x), error])

        if f(x) == 0 or (error is not None and error < tol):
            return x, error, i, iterations

        if f(a) * f(x) < 0:
            b = x
        else:
            a = x

        x_old = x

    return x, error, max_iter, iterations


def secant(f, x0, x1, tol, max_iter):
    iterations = []

    for i in range(1, max_iter + 1):
        if f(x1) - f(x0) == 0:
            break

        x = x1 - f(x1) * ((x1 - x0) / (f(x1) - f(x0)))
        error = abs(x - x1)

        iterations.append([i, x0, x1, x, f(x), error])

        if error < tol:
            return x, error, i, iterations

        x0, x1 = x1, x

    return x, error, max_iter, iterations


def newton_raphson(f, x0, tol, max_iter):
    iterations = []

    for i in range(1, max_iter + 1):
        f_x = f(x0)
        f_prime = derivative(f, x0)

        if f_prime == 0:
            break

        x1 = x0 - f_x / f_prime
        error = abs(x1 - x0)

        iterations.append([i, x0, f_x, error])

        if error < tol:
            return x1, error, i, iterations

        x0 = x1

    return x1, error, max_iter, iterations


def modified_secant(f, x0, delta, tol, max_iter):
    iterations = []

    for i in range(1, max_iter + 1):
        d = delta * x0

        if f(x0 + d) - f(x0) == 0:
            break

        x1 = x0 - f(x0) * (d / (f(x0 + d) - f(x0)))
        error = abs(x1 - x0)

        iterations.append([i, x0, f(x0), error])

        if error < tol:
            return x1, error, i, iterations

        x0 = x1

    return x1, error, max_iter, iterations


# ======================
# Flask Routes
# ======================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/solve", methods=["POST"])
def solve():
    root = None
    error = None
    its = None
    logs = []
    degree = int(request.form["degree"])
    coeffs = []

    for i in range(degree):
        coeffs.append(float(request.form[f"c{i}"]))

    const = float(request.form["const"])
    f = parse_function(coeffs, const)

    method = request.form["method"]
    tol = float(request.form["tol"])
    max_iter = int(request.form["iter"])

    # Method-specific inputs
    if method == "bisection":
        a = request.form.get("a", type=float)
        b = request.form.get("b", type=float)

        if a is None or b is None:
            return "Error: Please provide valid a and b values."


    elif method == "regula":
        a = request.form.get("a", type=float)
        b = request.form.get("b", type=float)

        if a is None or b is None:
            return "Error: Please provide valid a and b values."


    elif method == "secant":
        x0 = float(request.form["x0"])
        x1 = float(request.form["x1"])
        root, error, its, logs = secant(f, x0, x1, tol, max_iter)

    elif method == "newton":
        x0 = float(request.form["x0"])
        root, error, its, logs = newton_raphson(f, x0, tol, max_iter)

    elif method == "modified":
        x0 = float(request.form["x0"])
        delta = float(request.form["delta"])
        root, error, its, logs = modified_secant(f, x0, delta, tol, max_iter)
    else:
        return "Error: Unknown method selected"

    return render_template("result.html",
                           root=root,
                           error=error,
                           iters=its,
                           logs=logs,
                           method=method)


if __name__ == "__main__":
    app.run(debug=True)
