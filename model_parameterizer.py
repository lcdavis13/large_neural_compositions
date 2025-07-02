import numpy as np
from introspection import call, construct
from scipy.optimize import minimize_scalar


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fit_quadratic_width(factory_fn, widths, depth=1):
    params = [count_params(factory_fn(depth, w)) for w in widths]
    coeffs = np.polyfit(widths, params, deg=2)
    return coeffs  # a, b, c


def fit_linear_depth(factory_fn, depths, width=1):
    params = [count_params(factory_fn(d, width)) for d in depths]
    coeffs = np.polyfit(depths, params, deg=1)
    return coeffs  # m, c


def solve_quadratic(coeffs, target):
    a, b, c = coeffs
    disc = b**2 - 4*a*(c - target)
    if disc < 0:
        return 1
    return max(1, (-b + np.sqrt(disc)) / (2*a))


def solve_linear(coeffs, target):
    m, c = coeffs
    if m == 0:
        return 1
    return max(1, (target - c) / m)


def newton_refine(factory_fn, fixed, target, var='width', init=None, max_iter=10, tol=1e-3):
    x = init
    for _ in range(max_iter):
        if var == 'width':
            f = lambda w: count_params(factory_fn(fixed, int(round(w)))) - target
        else:  # var == 'depth'
            f = lambda d: count_params(factory_fn(int(round(d)), fixed)) - target

        fx = f(x)
        if abs(fx) < tol:
            break

        h = 1e-2 * x
        f_prime = (f(x + h) - f(x - h)) / (2 * h)
        if f_prime == 0:
            break

        x -= fx / f_prime
        x = max(1, x)

    return int(round(x))


def logspace_allocate_on_fixed_param_surface(w_est, d_est, R):
    C = w_est**2 * d_est  # approximate invariant
    log_w = R * np.log(w_est)  # interpolate from log(1) to log(w_est)
    w = np.exp(log_w)
    d = C / (w ** 2)
    return int(round(d)), int(round(w))


def refine_width_given_depth(factory_fn, depth, P_target, init_width, tol=1e-3):
    def objective(w):
        w_int = max(1, int(round(w)))
        model = factory_fn(depth, w_int)
        return abs(count_params(model) - P_target)

    res = minimize_scalar(objective, bounds=(init_width // 2, init_width * 2), method='bounded', options={'xatol': tol})
    return int(round(res.x))


def choose_depth_and_width(R, P_target, factory_fn,
                           depth_fit_range=[1, 2, 3, 4],
                           width_fit_range=[32, 64, 128, 256]):
    # Step 1: Fit curves
    quad_coeffs = fit_quadratic_width(factory_fn, width_fit_range, depth=1)
    lin_coeffs = fit_linear_depth(factory_fn, depth_fit_range, width=1)

    # Step 2: Estimate initial solutions
    w0 = solve_quadratic(quad_coeffs, P_target)
    d0 = solve_linear(lin_coeffs, P_target)

    # Step 3: Refine using Newton's method
    w_refined = newton_refine(factory_fn, fixed=1, target=P_target, var='width', init=w0)
    d_refined = newton_refine(factory_fn, fixed=1, target=P_target, var='depth', init=d0)

    # Step 4: Interpolate using a fair exchange of depth vs width via log parameterization
    depth_final, width_guess = logspace_allocate_on_fixed_param_surface(w_refined, d_refined, R)

    # Step 5: Refine width while holding depth fixed
    width_final = refine_width_given_depth(factory_fn, depth_final, P_target, width_guess)

    return {
        "depth": depth_final,
        "width": width_final,
    }


def load_model_2d(factory_fn, parameter_target, R, args):
    def wrapped_factory_fn(depth, width):
        return call(factory_fn, args, {"depth": depth, "width": width})[0]  # discard the override dictionary

    w_and_d = choose_depth_and_width(
        R=R,
        P_target=parameter_target,
        factory_fn=wrapped_factory_fn
    )

    model, override = call(factory_fn, args, w_and_d)
    return model, override


def choose_width(P_target, factory_fn, width_fit_range=[32, 64, 128, 256]):
    # Step 1: Fit quadratic curve for width (depth=1 as dummy)
    quad_coeffs = fit_quadratic_width(factory_fn, width_fit_range, depth=1)

    # Step 2: Solve for estimated width
    w0 = solve_quadratic(quad_coeffs, P_target)

    # Step 3: Refine width using Newton's method
    w_refined = newton_refine(factory_fn, fixed=1, target=P_target, var='width', init=w0)

    return {
        "width": w_refined,
    }


def load_model_1d(factory_fn, parameter_target, args):
    def wrapped_factory_fn(depth_unused, width):
        model, _ = call(factory_fn, args, {"width": width})
        return model

    w_only = choose_width(
        P_target=parameter_target,
        factory_fn=wrapped_factory_fn
    )

    model, override = call(factory_fn, args, w_only)
    return model, override

