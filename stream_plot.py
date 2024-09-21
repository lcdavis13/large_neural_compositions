import matplotlib.pyplot as plt
import itertools


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
    
    def plot(self, title, xlabel, ylabel, line_labels, x_value, y_values, add_point=False, x_log=False, y_log=False):
        plt.ion()
        
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
        
        plt.draw()
        plt.pause(0.00000001)
    
    def plot_point(self, title, line_label, x_value, y_value, symbol="o"):
        if title not in self.figs:
            print(f"Plot with title '{title}' does not exist.")
            return
        
        fig, ax = self.figs[title]
        lines = {line.get_label(): line for line in ax.get_lines()}
        
        if line_label in lines:
            line = lines[line_label]
            ax.text(x_value, y_value, '*', fontsize=18, color=line.get_color(), ha='center', va='center')
        else:
            print(f"Line with label '{line_label}' does not exist in the plot '{title}'.")
        
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.00000001)
    
    def plot_single(self, title, xlabel, ylabel, line_label, x_value, y_value, add_point=False, x_log=False,
                    y_log=False):
        self.plot(title, xlabel, ylabel, [line_label], x_value, [y_value], add_point=add_point, x_log=x_log,
                  y_log=y_log)
    
    def plot_loss(self, title, label, epoch, train_loss, validation_loss=None, add_point=False):
        labels = [f'{label} - Val Loss', f'{label} - Trn Loss']
        values = [validation_loss, train_loss]
        self.plot(title, 'Epoch', 'Loss', labels, epoch, values, add_point)
    
    def keep_plots_open(self):
        while plt.get_fignums():
            plt.waitforbuttonpress()


plotstream = PlotStreamer()


if __name__ == "__main__":
    plotstream.plot_loss("Training Progress", "Model A", 1, 0.5, None)
    plotstream.plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 10, [10, 5], None)
    plotstream.plot_loss("Training Progress", "Model A", 2, 0.4, 2.5, True)
    plotstream.plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 15, [8, 6], None)
    plt.pause(5.0)
    plotstream.plot_loss("Training Progress", "Model A", 3, 1.4, 0.5)
    plotstream.plot("other thing", "xthing", "ything", ["A thing 1", "A thing 2"], 20, [11, 9], None)
    plotstream.plot_loss("Training Progress", "Model B", 1, 1.5, None)
    plotstream.plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 10, [11, 6], None)
    plotstream.plot_loss("Training Progress", "Model B", 2, 1.4, 3.5, True)
    plotstream.plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 15, [9, 7], None)
    plotstream.plot_loss("Training Progress", "Model B", 3, 2.4, 1.5)
    plotstream.plot("other thing", "xthing", "ything", ["B thing 1", "B thing 2"], 20, [12, 10], None)
    
    print("done")
    plotstream.keep_plots_open()
