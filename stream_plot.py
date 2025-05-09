import multiprocessing as mp
import matplotlib.pyplot as plt
import time
import random
import signal
import sys
import atexit
import itertools
import matplotlib
from collections import defaultdict
from multiprocessing.queues import Queue as MpQueue




from collections import defaultdict
import matplotlib.pyplot as plt
import time

_last_inline_render_time = {}
_inline_plot_data = defaultdict(lambda: defaultdict(lambda: ([], [], None)))
_inline_plot_figures = {}  # Keeps figure/axis references per plot


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
_PLOT_MODE = "window" # Plot mode: 'window' (pop-up), 'inline' (in-process render), 'off' (disabled)
_PLOT_WAIT_ON_EXIT = False



def set_plot_mode(mode: str, wait_on_exit: bool = False):
    global _PLOT_MODE, _PLOT_WAIT_ON_EXIT
    assert mode in ("window", "inline", "off"), f"Invalid plot mode: {mode}"
    _PLOT_MODE = mode
    _PLOT_WAIT_ON_EXIT = wait_on_exit



def get_plot_mode():
    return _PLOT_MODE


def _ensure_mp_start_method():
    if get_plot_mode() == 'off':
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
    if get_plot_mode() == "off":
        return

    if get_plot_mode() == "inline":
        if plot_name not in _plot_queues:
            _plot_queues[plot_name] = []  # Inline uses simple list
        return

    # window mode (multiprocessing)
    _ensure_mp_start_method()
    global _plot_started, _shutdown_registered

    if plot_name not in _plot_queues or not isinstance(_plot_queues[plot_name], MpQueue):
        # Create queue and process
        pending = _plot_queues.get(plot_name, [])  # get any pending points (or empty)
        q = mp.Queue()
        _plot_queues[plot_name] = q

        config = _plot_configs.get(plot_name, {})
        print(f"[DEBUG] Creating plot process for '{plot_name}' with config: {config}")
        p = mp.Process(target=_live_plot, args=(plot_name, q, config, _PLOT_WAIT_ON_EXIT))
        p.start()
        _plot_processes[plot_name] = p

        # Flush pending points into real queue
        for item in pending:
            q.put(item)

        if not _shutdown_registered:
            _register_shutdown()
            _shutdown_registered = True

        _plot_started = True




def _register_shutdown():
    # Ensure shutdown happens automatically
    def shutdown_handler(*args):
        finish_up()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)
    atexit.register(finish_up)


def _live_plot(plot_name, q, config, wait_on_exit):
    if get_plot_mode() == 'off':
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

                plot_config = item.get('plot_config')
                if plot_config:
                    ax.set_title(plot_config.get('title', plot_name))
                    ax.set_xlabel(plot_config.get('xlabel', ''))
                    ax.set_ylabel(plot_config.get('ylabel', ''))

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
    fig.canvas.draw()
    plt.pause(0.001)

    if wait_on_exit:
        plt.show()  # Manual wait
    else:
        plt.close(fig)  # Auto-close



def _render_inline_plot(plot_name):
    # Create or reuse figure
    if plot_name not in _inline_plot_figures:
        fig, ax = plt.subplots()
        _inline_plot_figures[plot_name] = (fig, ax)
    else:
        fig, ax = _inline_plot_figures[plot_name]
        ax.clear()

    # Always apply config (even if no new points)
    config = _plot_configs.get(plot_name, {})
    ax.set_title(config.get("title", plot_name))
    ax.set_xlabel(config.get("xlabel", ""))
    ax.set_ylabel(config.get("ylabel", ""))
    if config.get("x_log", False):
        ax.set_xscale("log")
    if config.get("y_log", False):
        ax.set_yscale("log")

    # Process new data and accumulate into x/y lists
    new_queue = []
    for item in _plot_queues[plot_name]:
        if isinstance(item, dict) and item.get('cmd') == 'add_horizontal_line':
            y = item['y_value']
            lbl = item['label']
            line_style = item.get('style', {})
            ax.axhline(y=y, label=lbl, **line_style)
            new_queue.append(item)
        else:
            line_name, data_points, style, plot_config = item
            xdata, ydata, prev_style = _inline_plot_data[plot_name][line_name]

            for x, y in data_points:
                xdata.append(x)
                ydata.append(y)

            _inline_plot_data[plot_name][line_name] = (xdata, ydata, style or prev_style)

            # Update config if plot_config is passed here
            if plot_config:
                _plot_configs[plot_name] = plot_config

    # Clear queue
    _plot_queues[plot_name] = new_queue

    # Plot all lines
    for line_name, (xdata, ydata, style) in _inline_plot_data[plot_name].items():
        style = dict(style or {})
        is_marker_only = style.pop('_marker_only', False)

        if is_marker_only:
            style.setdefault('color', 'black')
            style.setdefault('marker', 'o')
            style.setdefault('linestyle', 'None')

        ax.plot(xdata, ydata, label=(line_name if not is_marker_only else None), **style)

    ax.legend()
    fig.canvas.draw()
    plt.pause(0.001)

    # NEW: Save the figure if requested
    if config.get("save_on_exit", False):
        filename = f"{plot_name}.png"
        fig.savefig(filename)
        print(f"[INFO] Saved inline plot '{plot_name}' to {filename}")


# === Public API ===

def plot_push(plot_name, line_name, data_point_or_list, style=None, *, plot_config=None):
    if get_plot_mode() == "off":
        return

    if isinstance(data_point_or_list, tuple):
        data_point_or_list = [data_point_or_list]

    if get_plot_mode() == "inline":
        if plot_config:
            _plot_configs[plot_name] = plot_config
        if plot_name not in _plot_queues:
            _plot_queues[plot_name] = []
        _plot_queues[plot_name].append((line_name, data_point_or_list, style, plot_config))
        return

    # window mode
    if plot_name not in _plot_queues:
        _plot_queues[plot_name] = []  # Start as a pending list

    if plot_config:
        if plot_name not in _plot_configs:
            _plot_configs[plot_name] = plot_config
        _ensure_plot_process(plot_name)  # Now we can start the process

    if isinstance(_plot_queues[plot_name], list):
        # Pending list mode (waiting for process)
        _plot_queues[plot_name].append((line_name, data_point_or_list, style))
    else:
        # Process already started
        queue = _plot_queues[plot_name]
        queue.put((line_name, data_point_or_list, style))



def finish_up(block=True):
    mode = get_plot_mode()

    if mode == 'off':
        return

    if mode == 'inline':
        print("[INFO] Final inline rendering of all plots...")
        for plot_name in _plot_queues.keys():
            _render_inline_plot(plot_name)
        if not _PLOT_WAIT_ON_EXIT:
            return
        input("[INFO] Press Enter to continue...")  # Allow user to view plot before exiting
        return

    if mode == 'window':
        if not _plot_started:
            return
        print("[INFO] Finalizing plots...")

        for q in _plot_queues.values():
            q.put(None)
        if block and _PLOT_WAIT_ON_EXIT:
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
    if get_plot_mode() == 'off':
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


def plot_horizontal_line(title, y_value, label, style=None):
    """
    Draw a horizontal line at y = y_value in the specified plot.
    Works for both 'inline' and 'window' modes.
    """
    if get_plot_mode() == 'off':
        return

    if style is None:
        style = {'linestyle': '--', 'color': 'gray'}

    cmd = {
        'cmd': 'add_horizontal_line',
        'label': label,
        'y_value': y_value,
        'style': style,
    }

    if get_plot_mode() == 'inline':
        if title not in _plot_queues:
            _plot_queues[title] = []
        _plot_queues[title].append(cmd)
        return

    # window mode
    if title not in _plot_queues:
        _plot_queues[title] = []  # Start as pending list
    if isinstance(_plot_queues[title], list):
        # Still pending stage, accumulate
        _plot_queues[title].append(cmd)
    else:
        # Process already started, send immediately
        queue = _plot_queues[title]
        queue.put(cmd)

def plot_horizontal_lines(title, line_dict, style=None):
    """
    Draw multiple horizontal lines in the specified plot.
    Works for both 'inline' and 'window' modes.
    """
    if get_plot_mode() == 'off':
        return

    if style is None:
        style = {'linestyle': '--', 'color': 'gray'}

    for label, y_value in line_dict.items():
        plot_horizontal_line(title, y_value, label, style=style)



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

    
    if get_plot_mode() == 'off':
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

    set_plot_mode('inline', wait_on_exit=False)
    data_producer()
    finish_up()
