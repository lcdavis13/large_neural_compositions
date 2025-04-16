import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import random
import signal
import sys
import atexit
import itertools

# Global per-plot color cycles

_color_list_standard = [
    # Tableau colors (default matplotlib "tab:" palette)
    'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple',
    'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',

    # Additional XKCD colors
    'xkcd:cerulean', 'xkcd:bright orange', 'xkcd:grass green',
    'xkcd:brick red', 'xkcd:deep violet', 'xkcd:chocolate brown',
    'xkcd:magenta', 'xkcd:slate grey', 'xkcd:olive green', 'xkcd:cyan',

    # Pastels
    'xkcd:pastel blue', 'xkcd:pastel orange', 'xkcd:pastel green',
    'xkcd:pastel red', 'xkcd:pastel purple', #'xkcd:pastel brown',
    'xkcd:pastel pink', 'xkcd:light grey', 'xkcd:pastel yellow',

    # Bonus vivid colors
    'xkcd:azure', 'xkcd:bright red', 'xkcd:chartreuse', 'xkcd:bright purple',
    'xkcd:teal', 'xkcd:coral', 'xkcd:royal blue', 'xkcd:neon green',
]



# === Internal plot manager state ===
_color_cycles_per_plot = {}
_plot_processes = {}
_plot_queues = {}
_plot_configs = {}
_loss_style_cache = {}  # {(title, label_prefix): color}
_plot_started = False
_shutdown_registered = False
_mp_start_method_set = False
_plot_refresh_interval = 0.5  # or whatever you prefer (e.g., 0.3 for ~3 FPS)
_plot_headless = False


def set_headless():
    """
    Set the plotting system to headless mode.
    This is useful for environments where no GUI is available (e.g., servers).
    """
    global _plot_headless
    _plot_headless = True


def _ensure_mp_start_method():
    if _plot_headless:
        return
    
    global _mp_start_method_set
    if not _mp_start_method_set:
        try:
            mp.set_start_method('spawn', force=True)  # force=True ensures it resets safely
            _mp_start_method_set = True
        except RuntimeError:
            # Start method was already set externally
            _mp_start_method_set = True



def _ensure_plot_process(plot_name):
    if _plot_headless:
        return
    
    _ensure_mp_start_method()
    global _plot_started, _shutdown_registered

    if plot_name not in _plot_queues:
        # Lazy creation of the queue and process
        q = mp.Queue()
        _plot_queues[plot_name] = q
        config = _plot_configs.get(plot_name, {})
        print(f"[DEBUG] Creating plot process for '{plot_name}' with config: {config}")
        p = mp.Process(target=_live_plot, args=(plot_name, q, config))
        p.start()
        _plot_processes[plot_name] = p

        if not _shutdown_registered:
            _register_shutdown()
            _shutdown_registered = True

        _plot_started = True


def _register_shutdown():
    # Ensure shutdown happens automatically
    def shutdown_handler(*args):
        wait_for_plot_exit()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(wait_for_plot_exit)


def _live_plot(plot_name, q, config):
    if _plot_headless:
        return
    
    plt.ion()
    fig, ax = plt.subplots()
    ax.set_title(config.get("title", plot_name))

    if config.get("x_log", False):
        ax.set_xscale('log')
    if config.get("y_log", False):
        ax.set_yscale('log')

    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))

    lines_data = {}   # {line_name: (xdata, ydata)}
    lines_plot = {}   # {line_name: line_plot_object}
    horizontal_lines = {}

    running = True
    
    while running:
        while not q.empty():
            item = q.get()
            if item is None:
                running = False
                break

            if isinstance(item, dict) and item.get('cmd') == 'add_horizontal_line':
                label = item['label']
                y_value = item['y_value']
                style = item.get('style', {'linestyle': '--', 'color': 'gray'})
                if label not in horizontal_lines:
                    line = ax.axhline(y=y_value, label=label, **style)
                    horizontal_lines[label] = line
                    ax.legend()
                continue

            line_name, data_points, style = item

            # Detect if this is a marker-only line (by convention)
            is_marker_only = style and style.get('_marker_only', False)

            if line_name not in lines_data:
                lines_data[line_name] = ([], [])
                style = style or {}
                is_marker_only = style.pop('_marker_only', False)  # ✅ Clean up internal flag

                if is_marker_only:
                    style.setdefault('color', 'black')
                    style.setdefault('marker', 'o')
                    style.setdefault('linestyle', 'None')

                line_plot, = ax.plot([], [], label=(line_name if not is_marker_only else None), **style)
                lines_plot[line_name] = line_plot

                if not is_marker_only:
                    ax.legend()


            xdata, ydata = lines_data[line_name]
            for x, y in data_points:
                xdata.append(x)
                ydata.append(y)
            lines_data[line_name] = (xdata, ydata)

            line_plot = lines_plot[line_name]
            line_plot.set_xdata(xdata)
            line_plot.set_ydata(ydata)

            ax.relim()
            ax.autoscale_view()

        plt.pause(_plot_refresh_interval)

    # Save plot if desired
    if config.get("save_on_exit", False):
        filename = f"{plot_name}.png"
        plt.savefig(filename)
        print(f"[INFO] Saved plot '{plot_name}' to {filename}")

    plt.ioff()
    plt.show()


# === Public API ===

def plot_push(plot_name, line_name, data_point_or_list, style=None, *, plot_config=None):
    """
    Push data to the plotting system.
    plot_name: name of the plot (will be created if not existing)
    line_name: name of the line within the plot
    data_point_or_list: (x, y) tuple or list of (x, y) tuples
    style: dict of line styles (optional)
    plot_config: dict of plot settings (optional, used only at first creation)
    """
    if _plot_headless:
        return
    
    if plot_name not in _plot_queues:
        if plot_config:
            _plot_configs[plot_name] = plot_config
        _ensure_plot_process(plot_name)

    queue = _plot_queues[plot_name]
    if isinstance(data_point_or_list, tuple):
        data_point_or_list = [data_point_or_list]

    queue.put((line_name, data_point_or_list, style))


def wait_for_plot_exit():
    """
    Gracefully shutdown all plot processes and wait for them to finish.
    """
    if _plot_headless:
        return
    
    global _plot_started
    if not _plot_started:
        return  # Nothing to do

    print("[INFO] Finalizing plots...")

    for q in _plot_queues.values():
        q.put(None)
    for p in _plot_processes.values():
        p.join()

    _plot_queues.clear()
    _plot_processes.clear()
    _plot_configs.clear()

    print("[INFO] All plots finalized.")

def _get_next_color_for_plot(plot_title):
    if plot_title not in _color_cycles_per_plot:
        _color_cycles_per_plot[plot_title] = itertools.cycle(_color_list_standard)
    return next(_color_cycles_per_plot[plot_title])


# === New Helper Function: plot() ===

def plot(title, xlabel, ylabel, line_labels, x_value, y_values, add_point=False, x_log=False, y_log=False):
    """
    High-level plotting function.
    """
    if _plot_headless:
        return
    
    assert len(line_labels) == len(y_values), "line_labels and y_values must have the same length"

    plot_config = {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'x_log': x_log,
        'y_log': y_log,
        'save_on_exit': True,
    }

    for label, y in zip(line_labels, y_values):
        # Auto-assign color for this line
        style_key = (title, label)
        if style_key not in _loss_style_cache:  # reuse the style cache
            assigned_color = _get_next_color_for_plot(title)
            _loss_style_cache[style_key] = assigned_color
        else:
            assigned_color = _loss_style_cache[style_key]

        style = {'color': assigned_color}

        # Main line
        plot_push(title, label, (x_value, y), style=style, plot_config=plot_config)

        if add_point:
            marker_label = f"{label}_point_markers"
            marker_style = {'_marker_only': True, 'marker': 'o', 'color': assigned_color}
            plot_push(title, marker_label, (x_value, y), style=marker_style, plot_config=plot_config)



# === New Helper Function: plot_horizontal_line() ===
def plot_horizontal_line(title, y_value, label, style=None):
    """
    Draw a true horizontal line at y = y_value in the specified plot.

    - title: str, name of the plot
    - y_value: float, the y-coordinate for the line
    - label: str, label of the line
    - style: dict of line style (optional)
    """
    if _plot_headless:
        return
    
    if style is None:
        style = {'linestyle': '--', 'color': 'gray'}

    if title not in _plot_queues:
        _ensure_plot_process(title)

    queue = _plot_queues[title]
    queue.put({
        'cmd': 'add_horizontal_line',
        'label': label,
        'y_value': y_value,
        'style': style
    })

def plot_single(title, xlabel, ylabel, line_label, x_value, y_value, add_point=False, x_log=False,
                y_log=False):
    """
    Helper method to call the plot function with only one value of y and its label.
    """
    plot(title, xlabel, ylabel, [line_label], x_value, [y_value], add_point=add_point, x_log=x_log, y_log=y_log)


def plot_loss(title, label_prefix, x, train_loss=None, validation_loss=None, xlabel="Epoch", ylabel="Loss", add_point=False):
    """
    Helper to plot training and validation loss curves with consistent styling.

    - title: plot title
    - label_prefix: shared label prefix
    - x: x-axis value
    - train_loss: float or None
    - validation_loss: float or None
    - add_point: bool, whether to add marker at current point
    """

    
    if _plot_headless:
        return
    
    plot_config = {
        'title': title,
        'xlabel': xlabel,
        'ylabel': ylabel,
        'x_log': False,
        'y_log': False,
        'save_on_exit': True,
    }

    # Step 1: Cache color per (plot, label_prefix)
    style_key = (title, label_prefix)

    if style_key not in _loss_style_cache:
        assigned_color = _get_next_color_for_plot(title)
        _loss_style_cache[style_key] = assigned_color
    else:
        assigned_color = _loss_style_cache[style_key]


    # Step 2: Prepare styles
    train_label = f"{label_prefix} - Trn"
    val_label = f"{label_prefix} - Val"

    train_style = {'color': assigned_color, 'linestyle': '--'}
    val_style = {'color': assigned_color, 'linestyle': '-'}

    # Step 3: Send data if provided
    if train_loss is not None:
        plot_push(title, train_label, (x, train_loss), style=train_style, plot_config=plot_config)

        if add_point:
            marker_label = f"{train_label}_point_markers"
            marker_style = {'_marker_only': True, 'marker': 'o', 'color': assigned_color}
            plot_push(title, marker_label, (x, train_loss), style=marker_style, plot_config=plot_config)

    if validation_loss is not None:
        plot_push(title, val_label, (x, validation_loss), style=val_style, plot_config=plot_config)

        if add_point:
            marker_label = f"{val_label}_point_markers"
            marker_style = {'_marker_only': True, 'marker': 'o', 'color': assigned_color}
            plot_push(title, marker_label, (x, validation_loss), style=marker_style, plot_config=plot_config)



# === Example usage ===

if __name__ == '__main__':

    def data_producer():
        for x in range(1, 101):
            # Dynamic values
            y_values_linear = [random.uniform(0.1, 10) * x, random.uniform(0.1, 5) * (100 - x)]
            y_values_linear_bigger = [random.uniform(0.1, 10) * x, 3.0*random.uniform(0.1, 5) * (100 - x)]
            y_values_log = [random.uniform(1, 10) * x, random.uniform(1, 5) * (100 - x + 1)]

            train_loss = random.uniform(0.5, 2) / x if x > 1 else None
            val_loss = random.uniform(0.3, 1.5) / x if x > 3 else None

            train_loss2 = random.uniform(0.5, 2) / (x-19) if x > 21 else None
            val_loss2 = random.uniform(0.3, 1.5) / (x-19) if x > 19 else None

            plot_loss(
                title='Training Curve',
                label_prefix='Model 1',
                x=x,
                train_loss=train_loss,
                validation_loss=val_loss,
                add_point=(x % 10 == 0)
            )

            
            # High-level plot function for linear
            plot(
                title='Dynamic Linear Plot',
                xlabel='X Axis',
                ylabel='Y Axis',
                line_labels=['line1', 'line2'],
                x_value=x,
                y_values=y_values_linear,
                add_point=(x % 10 == 0),
                x_log=False,
                y_log=False
            )

            # High-level plot function for log
            plot(
                title='Dynamic Logarithmic Plot',
                xlabel='X Axis',
                ylabel='Y Axis',
                line_labels=['lineA', 'lineB'],
                x_value=x,
                y_values=y_values_log,
                add_point=(x % 10 == 0),
                x_log=False,
                y_log=True
            )

            if x > 20:
                plot_loss(
                    title='Training Curve',
                    label_prefix='Model 2',
                    x=x,
                    train_loss=train_loss2,
                    validation_loss=val_loss2,
                    add_point=(x % 10 == 0)
                )

                plot(
                    title='Dynamic Linear Plot',
                    xlabel='X Axis',
                    ylabel='Y Axis',
                    line_labels=['line3', 'line4'],
                    x_value=x,
                    y_values=y_values_linear_bigger,
                    add_point=(x % 10 == 0),
                    x_log=False,
                    y_log=False
                )

            if x == 50:
                # Example: horizontal reference line
                plot_horizontal_line('Dynamic Linear Plot', y_value=200, label='Threshold')
                plot_horizontal_line('Dynamic Logarithmic Plot', y_value=500, label='Reference')

            time.sleep(0.1)

    data_producer()
    wait_for_plot_exit()
