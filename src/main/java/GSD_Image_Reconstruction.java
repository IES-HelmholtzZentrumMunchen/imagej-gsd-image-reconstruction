import ij.*;
import ij.plugin.filter.PlugInFilter;
import ij.process.ImageProcessor;
import ij.gui.*;

import java.util.ArrayList;
import java.util.concurrent.*;

/**
 * GSD image reconstruction plugin
 *
 * This plugin can be used to reconstruct GSD images.
 *
 * @author Julien Pontabry
 */
public class GSD_Image_Reconstruction implements PlugInFilter
{
    /**
     * Title of the plugin used on top of dialogs.
     */
    private String m_pluginTitle = "GSD image reconstruction";

    /**
     * Kernel bandwidth used for the kernel smoothing method.
     */
    private double m_kernelBandwidth = 5.0;

    /**
     * Input GSD image.
     */
    private ImagePlus m_inputImage = null;

    /**
     * Minimal and maximal intensity values (for normalisation).
     */
    private double m_minIntensity = 100000;
    private double m_maxIntensity = 0;

    /**
     * Current and maximal progress of progress bar.
     */
    private int m_currentProgress = 0;
    private int m_maxProgress = 0;


	/**
	 * @see ij.plugin.filter.PlugInFilter#setup(java.lang.String, ij.ImagePlus)
	 */
	@Override
	public int setup(String arg, ImagePlus imp) {
        // Create GUI
        GenericDialog gui = new GenericDialog(m_pluginTitle);
        gui.addNumericField("kernel bandwidth", m_kernelBandwidth, 1);
        gui.showDialog();

        // Check GUI events
        if(gui.wasCanceled())
        {
            return DONE;
        }

        // Get parameters
        m_kernelBandwidth = gui.getNextNumber();

        // Get input image
        m_inputImage = imp;

        // Return process
		return DOES_16+NO_CHANGES;
	}

	/**
	 * @see ij.plugin.filter.PlugInFilter#run(ij.process.ImageProcessor)
	 */
	@Override
	public void run(ImageProcessor ip) {
        ImagePlus outputImage = this.reconstruct(m_inputImage);
        outputImage.show();
	}

    public ImagePlus reconstruct(ImagePlus inputImage) {
        // Allocate memory for the output
        ImagePlus outputImage = NewImage.createImage("reconstructed image", inputImage.getWidth(), inputImage.getHeight(), inputImage.getNSlices(), inputImage.getBitDepth(), NewImage.FILL_BLACK);
        outputImage.copyScale(inputImage);

        // Get processors
        ImageProcessor inputProcessor = inputImage.getProcessor();
        ImageProcessor outputProcessor = outputImage.getProcessor();

        // Precompute constants
        double bandwidthSquare = m_kernelBandwidth * m_kernelBandwidth;
        int maxPixelDistance = (int)Math.round(m_kernelBandwidth*5);

        // First, we estimate the distribution using double precision
        double[][] tmp = new double[outputImage.getWidth()][outputImage.getHeight()];

        // Display a progress bar
        m_maxProgress = outputImage.getWidth()*outputImage.getHeight();
        IJ.showProgress(m_currentProgress, m_maxProgress);

        ArrayList< ParallelInput > inputs = new ArrayList< ParallelInput >();

        for(int y = 0; y < outputImage.getHeight(); y++) {
            for(int x = 0; x < outputImage.getWidth(); x++) {
                ParallelInput input = new ParallelInput();
                input.x = x;
                input.y = y;
                input.tmp = tmp;
                input.inputProcessor = inputProcessor;
                input.bandwidthSquare = bandwidthSquare;
                input.maxPixelDistance = maxPixelDistance;

                inputs.add(input);
            }
        }

        try {
            processInputs(inputs);
        } catch (Exception e){
            IJ.showMessage(e.getMessage());
        }

        // Normalise the intensity values
        for(int y = 0; y < outputImage.getHeight(); y++) {
            for(int x = 0; x < outputImage.getWidth(); x++) {
                outputProcessor.set(x,y,(int)Math.round(65535*(tmp[x][y]- m_minIntensity)/(m_maxIntensity - m_minIntensity)));
            }
        }

        return outputImage;
    }

    /**
     * Simple for loop parallelisation.
     * @param inputs Input parameters (local and not shared)
     * @return Output parameters
     * @throws InterruptedException
     * @throws ExecutionException
     */
    private ArrayList< ParallelOutput > processInputs(ArrayList< ParallelInput > inputs) throws InterruptedException, ExecutionException {
        // Create a service for threads (same number as available processors)
        int threads = Runtime.getRuntime().availableProcessors() + 1;
        ExecutorService service = Executors.newFixedThreadPool(threads);

        // Create a list of future outputs
        ArrayList< Future< ParallelOutput > > futures = new ArrayList< Future< ParallelOutput > >();

        // For each input, do a parallel process using service
        for(final ParallelInput input : inputs) {
            // Create a callable defining the process
            final Callable< ParallelOutput > callable = new Callable< ParallelOutput >() {
                public ParallelOutput call() throws Exception {
                    ParallelOutput output = new ParallelOutput();

                    // Precompute the block for the neighborhood
                    int minXi = input.x-input.maxPixelDistance; if(minXi < 0) minXi = 0;
                    int maxXi = input.x+input.maxPixelDistance; if(maxXi >= input.inputProcessor.getWidth()) maxXi = input.inputProcessor.getWidth()-1;
                    int minYi = input.y-input.maxPixelDistance; if(minYi < 0) minYi = 0;
                    int maxYi = input.y+input.maxPixelDistance; if(maxYi >= input.inputProcessor.getHeight()) maxYi = input.inputProcessor.getHeight()-1;

                    // Estimate the density at (x,y)
                    for(int x_i = minXi; x_i < maxXi; x_i++) {
                        for(int y_i = minYi; y_i < maxYi; y_i++) {
                            int nbOfEvents = input.inputProcessor.get(x_i, y_i);

                            if(nbOfEvents > 0) { // If there is at least one event
                                double xdiff = (input.x - x_i);
                                double ydiff = (input.y - y_i);

                                input.tmp[input.x][input.y] += Math.exp(-0.5 * (xdiff*xdiff + ydiff*ydiff) / input.bandwidthSquare) * nbOfEvents;
                            }
                        }
                    }

                    // Normalise to prevent loss of intensity on the borders
                    input.tmp[input.x][input.y] /= (maxXi-minXi) * (maxYi-minYi);

                    synchronized (this) {
                        // Search for min and max intensity values
                        if(input.tmp[input.x][input.y] > m_maxIntensity) m_maxIntensity = input.tmp[input.x][input.y];
                        if(input.tmp[input.x][input.y] < m_minIntensity) m_minIntensity = input.tmp[input.x][input.y];

                        // Update progress bar
                        IJ.showProgress(++m_currentProgress, m_maxProgress);
                    }

                    return output;
                }
            };

            // Add future results as outputs
            futures.add(service.submit(callable));
        }

        // Stop the threads service
        service.shutdown();

        // Create an output list
        ArrayList< ParallelOutput > outputs = new ArrayList< ParallelOutput >();

        // Fill the list with the results
        for(Future<ParallelOutput> future : futures) {
            outputs.add(future.get());
        }

        return outputs;
    }

    /**
     * Input parameters container for parallel processing.
     */
    private class ParallelInput {
        public int x;
        public int y;
        public double[][] tmp;
        public ImageProcessor inputProcessor;
        public double bandwidthSquare;
        public int maxPixelDistance;
    }

    /**
     * Output parameters container for parallel processing.
     * Note that this is empty because the program does not need to produce output directly.
     */
    private class ParallelOutput {
        // ----
    }

	/**
	 * Main method for debugging.
	 *
	 * For debugging, it is convenient to have a method that starts ImageJ, loads an
	 * image and calls the plugin, e.g. after setting breakpoints.
	 *
	 * @param args unused
	 */
	public static void main(String[] args) {
		// set the plugins.dir property to make the plugin appear in the Plugins menu
		Class<?> clazz = GSD_Image_Reconstruction.class;
		String url = clazz.getResource("/" + clazz.getName().replace('.', '/') + ".class").toString();
		String pluginsDir = url.substring(5, url.length() - clazz.getName().length() - 6);
		System.setProperty("plugins.dir", pluginsDir);

		// start ImageJ
		new ImageJ();
	}
}
