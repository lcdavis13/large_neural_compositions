import matplotlib.pyplot as plt
import itertools
import threading
import queue
import time
import matplotlib

# Use a backend that supports threading (e.g., TkAgg, Qt5Agg)
matplotlib.use('TkAgg')


class PlotStreamer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        self.figs = {}
        self.label_styles = {}
        self.queue = queue.Queue()  # Queue to handle data between threads
        self.plotting_thread = threading.Thread(target=self._plotting_loop, daemon=True)
        self.plotting_thread.start()
    
    def plot(self, title, xlabel, ylabel, line_labels, x_value, y_values, add_point=False, x_log=False, y_log=False):
        # Add plotting instructions to the queue
        self.queue.put((title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log))
    
    def _plotting_loop(self):
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
                    title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log = data
                    self._update_data(title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log)
                
                # Now update the plot after processing all data in the queue
                self._draw_plots()
            
            # Keep figures responsive
            plt.pause(0.001)
    
    def _update_data(self, title, xlabel, ylabel, line_labels, x_value, y_values, add_point, x_log, y_log):
        if title in self.figs and not plt.fignum_exists(self.figs[title][0].number):
            self.figs[title] = None
            return
        
        if title not in self.figs:
            fig, ax = plt.subplots()
            self.figs[title] = (fig, ax)
        
        fig, ax = self.figs[title]
        num_existing_lines = len(ax.get_lines())
        num_new_lines = len(line_labels)
        color_index = num_existing_lines // num_new_lines
        
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        current_color = color_cycle[color_index % len(color_cycle)]
        
        style_cycle = itertools.cycle(['-', '--', '-.', ':'])
        
        for i, (label, value) in enumerate(zip(line_labels, y_values)):
            if label not in self.label_styles:
                self.label_styles[label] = next(itertools.islice(style_cycle, i, None))
        
        lines = {line.get_label(): line for line in ax.get_lines()}
        
        for label, value in zip(line_labels, y_values):
            linestyle = self.label_styles[label]
            if label in lines:
                line = lines[label]
                if value is not None:
                    line.set_xdata(list(line.get_xdata()) + [x_value])
                    line.set_ydata(list(line.get_ydata()) + [value])
                    if add_point:
                        ax.text(x_value, value, '*', fontsize=18, color=line.get_color(), ha='center', va='center')
            else:
                if value is not None:
                    ax.plot([x_value], [value], label=label, color=current_color, linestyle=linestyle)
                    if add_point:
                        ax.text(x_value, value, '*', fontsize=18, color=current_color, ha='center', va='center')
        
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
        """Draw the plots after processing all queued data."""
        plt.draw()
    
    def keep_plots_open(self):
        """Block the main thread until all plot windows are closed."""
        while plt.get_fignums():  # Check if any figures are open
            time.sleep(0.1)
    
    # ----------------- Convenience Methods ----------------- #
    def plot_point(self, title, line_label, x_value, y_value, symbol="o"):
        """
        Plots a single point on an existing curve.
        """
        self.plot(title, "", "", [line_label], x_value, [y_value], add_point=True)
    
    def plot_single(self, title, xlabel, ylabel, line_label, x_value, y_value, add_point=False, x_log=False,
                    y_log=False):
        """
        Helper method to call the plot function with only one value of y and its label.
        """
        self.plot(title, xlabel, ylabel, [line_label], x_value, [y_value], add_point=add_point, x_log=x_log,
                  y_log=y_log)
    
    def plot_loss(self, title, label, epoch, train_loss, validation_loss=None, add_point=False):
        """
        Helper function to plot validation and training loss curves.
        """
        labels = [f'{label} - Val Loss', f'{label} - Trn Loss']
        values = [validation_loss, train_loss]
        self.plot(title, 'Epoch', 'Loss', labels, epoch, values, add_point)


# Create a global instance of the PlotManager
plotstream = PlotStreamer()


if __name__ == "__main__":
    for i in range(10):
        time.sleep(0.5)  # Simulate a long computation
        plotstream.plot_loss("Training Progress", "Model A", i, train_loss=i ** 2, validation_loss=i ** 1.5)

    print("Computation complete.")

    plotstream.keep_plots_open()
