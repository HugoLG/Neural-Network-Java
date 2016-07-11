import java.util.Vector;

/**
 * FeedForward with Backpropagation Artificial Neural Network
 * This class can be use to create a FeedForward with Backpropagation
 * neural network, the parameters values of the network are customizable,
 * also the number of neurons and the number of hidden layers are defined
 * by the user when creating the neural network object.
 *
 * <b> Note: </b> This class needs the Neuron class.
 *
 * @author Hugo León Garza
 * @version 1.0
 * @since July-2016
 *
 */

public class NeuralNetwork {

    private double lambda;
    private double eta;
    private double momentum;
    private Vector<Vector<Neuron>> neuralNetwork;
    private int numInputNeurons;
    private int numOutputNeurons;
    private int numHiddenNeurons;

    //Constructor
    public NeuralNetwork(Double l, Double e, Double a, int inNeurons, int numHiddenLayers, int hiddenNeurons, int outNeurons, boolean addHiddenBias) {
        this.lambda = l;
        this.eta = e;
        this.momentum = a;
        this.numInputNeurons = inNeurons;
        this.numOutputNeurons = outNeurons;
        this.numHiddenNeurons = hiddenNeurons;
        this.neuralNetwork = new Vector<Vector<Neuron>>();

        Vector<Neuron> inputLayer = new Vector<Neuron>();
        for (int i=0; i<inNeurons; i++) {
            Neuron n = new Neuron(numHiddenNeurons);
            inputLayer.add(n);
        }
        neuralNetwork.add(inputLayer);

        Vector<Neuron> hiddenLayer = new Vector<Neuron>();
        if(addHiddenBias) {
            hiddenLayer.add(new Neuron(true));
        }
        for(int i=0; i<numHiddenLayers; i++) {
            for(int j=0; j<hiddenNeurons; j++) {
                Neuron n;
                if((j+1) < hiddenNeurons) {
                    n = new Neuron(numHiddenNeurons);
                } else {
                    n = new Neuron(numOutputNeurons);
                }
                hiddenLayer.add(n);
            }
            neuralNetwork.add(hiddenLayer);
        }

        Vector<Neuron> outputLayer = new Vector<Neuron>();
        for (int i=0; i<outNeurons; i++) {
            Neuron n = new Neuron();
            outputLayer.add(n);
        }
        neuralNetwork.add(outputLayer);
    }

    //METHODS
    public void printNetwork() {
        System.out.println("Lambda: " + this.lambda);
        System.out.println("Eta: " + this.eta);
        System.out.println("Momentum: " + this.momentum);
        System.out.println("Neural Network... ");
        System.out.println("Input Layer: ");
        for(int i=0; i<neuralNetwork.get(0).size(); i++) {
            neuralNetwork.get(0).get(i).print();
            System.out.print(" ");
        }
        System.out.println();
        System.out.println("Hidden Layers: ");
        for(int i=1; i<neuralNetwork.size()-1; i++) {
            for(int j=0; j<neuralNetwork.get(i).size(); j++) {
                neuralNetwork.get(i).get(j).print();
                System.out.print(" ");
            }
            System.out.println();
        }
        System.out.println("Output Layer: ");
        for(int i=0; i<neuralNetwork.get(neuralNetwork.size()-1).size(); i++) {
            neuralNetwork.get(neuralNetwork.size()-1).get(i).print();
            System.out.print(" ");
        }
        System.out.println();
    }

    public void trainNetwork(String trainingFilename) {
        //read training examples
        //send each example to forward and backpropagation
        //feedforward
        //calculate output
        //calculate error;
        //backpropagation
        //calculate local gradient
        //calculate change of weights
        //update weights
        //
        //calculate training epoch error;
        //validation forward
        //calculate validation error;
    }

    /**
     * This method is used to make the Neural Network go through the
     * Feed Forward process one time with the training sample that was
     * received as parameter.
     *
     * @param trainingSample This is a vector with the input values for each neuron. (Values must be a Double)
     */
    public void feedForward(Vector<Double> trainingSample) {
        for(int i=0; i<trainingSample.size(); i++) {
            neuralNetwork.get(0).get(i).setValue(trainingSample.get(i));
        }

        //All layers will use the same activation function, if a layer uses another activation function
        //modifications to this for has to be made
        for(int i=1; i<neuralNetwork.size(); i++) {

            for(int j=0; j<neuralNetwork.get(i).size(); j++) {
                double tmpV = 0;
                for(int k=0; k<neuralNetwork.get(i-1).size(); k++) {
                    tmpV += (neuralNetwork.get(i-1).get(k).getValue() * neuralNetwork.get(i-1).get(k).getWeight(j));
                }
                neuralNetwork.get(i).get(j).setValue(tmpV);
                neuralNetwork.get(i).get(j).calculateH(1, this.lambda);
            }
        }
    }

    /**
     *
     *
     */
    public Vector<Double> calculateErrors(Vector<Double> target) {

        Vector<Double> tmp = new Vector<Double>();

        for(int i=0; i<target.size(); i++) {
            tmp.add(target.get(i) - neuralNetwork.get(neuralNetwork.size()-1).get(i).getValue());
        }

        return tmp;

    }

    /**
     *
     *
     *
     */
    public void backPropagation(Vector<Double> errors) {

    }

    //SETTERS
    public void setLambda(double l) {
        this.lambda = l;
    }

    public void setMomentum(double a) {
        this.momentum = a;
    }

    public void setEta(double e) {
        this.eta = e;
    }

    public void setNeuralNetwork(Vector<Vector<Neuron>> nn) {
        this.neuralNetwork = nn;
    }

    public void setNumInputNeurons(int i) {
        this.numInputNeurons = i;
    }

    public void setNumOutputNeurons(int i) {
        this.numOutputNeurons = i;
    }

    public void setNumHiddenNeurons(int i) {
        this.numHiddenNeurons = i;
    }

    //GETTERS
    public double getLambda() {
        return this.lambda;
    }

    public double getMomentum() {
        return this.momentum;
    }

    public double getEta() {
        return this.eta;
    }

    public Vector<Vector<Neuron>> getNeuralNetwork() {
        return this.neuralNetwork;
    }

    public int getNumInputNeurons() {
        return this.numInputNeurons;
    }

    public int getNumOutputNeurons() {
        return this.numOutputNeurons;
    }

    public int getNumHiddenNeurons() {
        return this.numHiddenNeurons;
    }
}
