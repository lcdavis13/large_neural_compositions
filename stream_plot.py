import matplotlib.pyplot as plt
import itertools
import threading
import queue
import time
import matplotlib

# Use a backend that supports threading (e.g., TkAgg, Qt5Agg)
# matplotlib.use('TkAgg')


class PlotStreamer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.headless = False
        self.figs = {}
        self.label_styles = {}
        self.color_cycle_positions = {}  # Dictionary to track color cycle position per plot
        self.queue = queue.Queue()  # Queue to handle data between threads
        self.plotting_thread = threading.Thread(target=self._plotting_loop, daemon=True)
        self.plotting_thread.start()
    
    def plot(self, title, xlabel, ylabel, line_labels, x_value, y_values, add_point=False, x_log=False, y_log=False,
             hline=None, vline=None):
        if self.headless:
            return
        """
        Add plotting instructions to the queue, including optional horizontal and vertical lines.

        :param title: The title of the plot.
        :param xlabel: The label for the X-axis.
        :param ylabel: The label for the Y-axis.
        :param line_labels: A list of labels for the plotted lines or lines (including hline/vline).
        :param x_value: The X-value for the point to be plotted.
        :param y_values: A list of Y-values corresponding to the labels.
        :param add_point: Whether to add a point marker to the plot.
        :param x_log: Whether the X-axis should be logarithmic.
        :param y_log: Whether the Y-axis should be logarithmic.
        :param hline: A tuple (y_value, label, color, linestyle) for horizontal lines. Default is None.
        :param vline: A tuple (x_value, label, color, linestyle) for vertical lines. Default is None.
        """
        # Pass hline and vline through the queue
        self.queue.put((title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log, hline, vline))
    
    def _plotting_loop(self):
        if self.headless:
            return
        while True:
            batch_data = []
            try:
                # Fetch all data points from the queue without blocking
                while True:
                    batch_data.append(self.queue.get_nowait())
            except queue.Empty:
                pass
            
            # Only plot if we have accumulated some data
            if batch_data:
                for data in batch_data:
                    # Process each data point in batch but do not plot after each one
                    # Unpacking new hline and vline arguments
                    title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log, hline, vline = data
                    self._update_data(title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log,
                                      hline, vline)
                
                # Now update the plot after processing all data in the queue
                self._draw_plots()
            
            # Keep figures responsive
            plt.pause(0.001)
    
    def _update_data(self, title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log, hline,
                     vline):
        if self.headless:
            return
        """
        Update data for the plots, including optional horizontal and vertical lines.
        """
        if title in self.figs and not plt.fignum_exists(self.figs[title][0].number):
            self.figs[title] = None
            return
        
        if title not in self.figs:
            fig, ax = plt.subplots()
            self.figs[title] = (fig, ax)
            self.color_cycle_positions[title] = 0  # Initialize color cycle position for new plot
        
        fig, ax = self.figs[title]
        
        # Handle regular lines
        if line_labels and y_values:
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            style_cycle = itertools.cycle(['-', '--', '-.', ':'])
            
            # Get the current color cycle position for this plot
            current_position = self.color_cycle_positions.get(title, 0)
            current_color = color_cycle[current_position % len(color_cycle)]
            
            lines = {line.get_label(): line for line in ax.get_lines()}
            
            # Track if any new lines are added
            new_lines_added = False
            
            for i, (label, value) in enumerate(zip(line_labels, y_values)):
                if label in lines:
                    # Update existing line if it has already been plotted
                    line = lines[label]
                    if value is not None:
                        line.set_xdata(list(line.get_xdata()) + [x_value])
                        line.set_ydata(list(line.get_ydata()) + [value])
                        if add_point:
                            ax.text(x_value, value, '*', fontsize=18, color=line.get_color(), ha='center', va='center')
                else:
                    # Use the same color for all new lines in this batch, determined by the saved position
                    color = current_color
                    linestyle = next(style_cycle)
                    
                    if label in self.label_styles:
                        linestyle = self.label_styles[label]
                    else:
                        self.label_styles[label] = linestyle
                    
                    # Register the line, even if value is None (plot it with NaN values to maintain style/color)
                    if value is not None:
                        ax.plot([x_value], [value], label=label, color=color, linestyle=linestyle)
                        if add_point:
                            ax.text(x_value, value, '*', fontsize=18, color=color, ha='center', va='center')
                    else:
                        # Create an empty line with NaN values to maintain the label and color for future points
                        ax.plot([x_value], [float('nan')], label=label, color=color, linestyle=linestyle)
                    
                    # Mark that new lines were added
                    new_lines_added = True
            
            # Increment the color cycle position only if new lines were added in this batch
            if new_lines_added:
                self.color_cycle_positions[title] = (current_position + 1) % len(color_cycle)
        
        # Handle horizontal lines
        if hline:
            y_value, hline_label, hline_color, hline_style = hline
            if line_labels and hline_label:
                hline_label = line_labels[0]
            ax.axhline(y=y_value, color=hline_color or 'black', linestyle=hline_style or '--', label=hline_label)
        
        # Handle vertical lines
        if vline:
            x_value, vline_label, vline_color, vline_style = vline
            if line_labels and vline_label:
                vline_label = line_labels[1] if len(line_labels) > 1 else line_labels[0]
            ax.axvline(x=x_value, color=vline_color or 'black', linestyle=vline_style or '--', label=vline_label)
        
        # Adjust the plot appearance
        ax.relim()
        ax.autoscale_view()
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()
        
        if x_log:
            ax.set_xscale('log')
        if y_log:
            ax.set_yscale('log')
    
    def _draw_plots(self):
        if self.headless:
            return
        """Draw the plots after processing all queued data."""
        plt.draw()
    
    def wait_for_plot_exit(self):
        if self.headless:
            return
        """Block the main thread until all plot windows are closed."""
        while plt.get_fignums():  # Check if any figures are open
            time.sleep(0.1)
    
    # ----------------- Convenience Methods ----------------- #
    
    def plot_loss(self, title, label, epoch, train_loss, validation_loss=None, add_point=False):
        """
        Helper function to plot validation and training loss curves.
        """
        labels = [f'{label} - Val Loss', f'{label} - Trn Loss']
        values = [validation_loss, train_loss]
        self.plot(title, 'Epoch', 'Loss', labels, epoch, values, add_point)
        
        
    def plot_point(self, title, line_label, x_value, y_value, symbol="o"):
        """
        Plots a single point on an existing curve.
        """
        self.plot(title, "", "", [line_label], x_value, [y_value], add_point=True)
    
    def plot_horizontal_line(self, title, y_value, label=None, color='black', linestyle='--'):
        """
        Helper method to add a horizontal line by calling plot with hline.

        :param title: The title of the plot.
        :param y_value: The y-coordinate for the horizontal line.
        :param label: The label for the horizontal line.
        :param color: The color of the horizontal line (default is black).
        :param linestyle: The style of the horizontal line (default is dashed).
        """
        hline = (y_value, label, color, linestyle)
        self.plot(title, "", "", [label], None, [], hline=hline)
    
    def plot_vertical_line(self, title, x_value, label=None, color='black', linestyle='--'):
        """
        Helper method to add a vertical line by calling plot with vline.

        :param title: The title of the plot.
        :param x_value: The x-coordinate for the vertical line.
        :param label: The label for the vertical line.
        :param color: The color of the vertical line (default is black).
        :param linestyle: The style of the vertical line (default is dashed).
        """
        vline = (x_value, label, color, linestyle)
        self.plot(title, "", "", [label], None, [], vline=vline)
    
    def plot_single(self, title, xlabel, ylabel, line_label, x_value, y_value, add_point=False, x_log=False,
                    y_log=False):
        """
        Helper method to call the plot function with only one value of y and its label.
        """
        self.plot(title, xlabel, ylabel, [line_label], x_value, [y_value], add_point=add_point, x_log=x_log,
                  y_log=y_log)
        
    def set_headless(self):
        """
        Set the backend to headless (non-interactive) for running on servers.
        """
        self.headless = True


# Create a global instance of the PlotManager
plotstream = PlotStreamer()


if __name__ == "__main__":
    plotstream.plot_horizontal_line("Training Progress", 10, "Target Loss")
    for i in range(10):
        time.sleep(0.5)  # Simulate a long computation
        plotstream.plot_loss("Training Progress", "Model A", i, train_loss=i ** 2, validation_loss=i ** 1.5)
        plotstream.plot_loss("Training Progress", "Model B", i, train_loss=i ** 2 - i, validation_loss=i ** 1.5 + 1)
    plotstream.plot_vertical_line("Training Progress", 5, "Halfway")
    print("Computation complete.")

    plotstream.wait_for_plot_exit()
