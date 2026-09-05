import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.*;
import java.io.*;
import java.util.*;
import java.util.List;

/**
 * wrote hte sing
 * Complete Single-File PPO (Proximal Policy Optimization) Mario-Style
 * Platformer in Java.
 * Includes:
 * 1. Deep Neural Network & Matrix Library from scratch (Dense, ReLU, Softmax,
 * Adam Optimizer).
 * 2. Actor-Critic PPO Reinforcement Learning Engine with Generalized Advantage
 * Estimation (GAE).
 * 3. Side-Scrolling 2D Mario Physics & Environment (Platforms, Pitfalls,
 * Enemies, Coins, Goal Flag).
 * 4. Rich Swing Visualizer Dashboard (Live Game Canvas, Policy Distribution,
 * Reward Chart, Model Save/Load).
 */
public class PPOMarioGame extends JFrame {

    public static final int GAME_WIDTH = 550;
    public static final int GAME_HEIGHT = 450;
    public static final int DASHBOARD_WIDTH = 450;
    public static final int LEVEL_WIDTH = 2600;

    // Core System Modules
    private MarioEnv env;
    private PPOAgent agent;

    // Simulation Flags and Performance Variables
    private boolean isHumanMode = false;
    private boolean isFastForward = false;
    private boolean isPaused = false;
    private int fastForwardSpeed = 25; // Iterations per frame loop in fast mode

    private int episodeCount = 0;
    private double currentEpisodeReward = 0.0;
    private int highScore = 0;
    private final List<Double> rewardHistory = new ArrayList<>();
    private final List<Double> avgRewardHistory = new ArrayList<>();
    private double lastActorLoss = 0.0;
    private double lastCriticLoss = 0.0;

    // Swing Visual Components
    private GameCanvas gamePanel;
    private GraphPanel graphPanel;
    private JLabel lblEpisode, lblScore, lblHighScore, lblReward, lblAvgReward, lblFPS, lblLoss, lblActionName;
    private JProgressBar[] barActionProbs;
    private JButton btnToggleMode, btnToggleSpeed, btnPause, btnReset, btnSave, btnLoad;

    // Loop Timing
    private javax.swing.Timer gameTimer;
    private long lastFrameTime = System.currentTimeMillis();
    private int fpsCounter = 0;
    private int currentFPS = 60;

    public PPOMarioGame() {
        super("PPO AI Mario Platformer - Deep Reinforcement Learning Engine");
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setResizable(false);
        setLayout(new BorderLayout());

        env = new MarioEnv(LEVEL_WIDTH, GAME_HEIGHT);
        // State dimension = 10, Action dimension = 5 (0: None, 1: Left, 2: Right, 3:
        // Jump, 4: Right+Jump)
        agent = new PPOAgent(10, 5);

        initUI();

        // 60 FPS Target Timer loop
        gameTimer = new javax.swing.Timer(16, e -> gameLoop());
        gameTimer.start();

        pack();
        setLocationRelativeTo(null);
        setVisible(true);
    }

    private void initUI() {
        JPanel mainContainer = new JPanel(new BorderLayout());
        mainContainer.setBackground(new Color(24, 26, 34));

        // Game Visual Canvas (Left)
        gamePanel = new GameCanvas(env);
        gamePanel.setPreferredSize(new Dimension(GAME_WIDTH, GAME_HEIGHT));
        mainContainer.add(gamePanel, BorderLayout.WEST);

        // Keyboard Controls for Human Gameplay
        gamePanel.setFocusable(true);
        gamePanel.addKeyListener(new KeyAdapter() {
            @Override
            public void keyPressed(KeyEvent e) {
                if (!isHumanMode)
                    return;
                int key = e.getKeyCode();
                if (key == KeyEvent.VK_LEFT || key == KeyEvent.VK_A)
                    env.setHumanAction(1);
                else if (key == KeyEvent.VK_RIGHT || key == KeyEvent.VK_D)
                    env.setHumanAction(2);
                else if (key == KeyEvent.VK_SPACE || key == KeyEvent.VK_W || key == KeyEvent.VK_UP)
                    env.setHumanAction(3);
                else if (key == KeyEvent.VK_SHIFT)
                    env.setHumanAction(4);
            }

            @Override
            public void keyReleased(KeyEvent e) {
                if (isHumanMode)
                    env.setHumanAction(0);
            }
        });

        // Dashboard Panel (Right)
        JPanel dashboard = new JPanel();
        dashboard.setLayout(new BoxLayout(dashboard, BoxLayout.Y_AXIS));
        dashboard.setPreferredSize(new Dimension(DASHBOARD_WIDTH, GAME_HEIGHT));
        dashboard.setBackground(new Color(18, 20, 26));
        dashboard.setBorder(new EmptyBorder(10, 10, 10, 10));

        // Title Banner
        JLabel title = new JLabel("PPO MARIO RL ENGINE");
        title.setFont(new Font("SansSerif", Font.BOLD, 18));
        title.setForeground(new Color(0, 215, 255));
        title.setAlignmentX(Component.CENTER_ALIGNMENT);
        dashboard.add(title);
        dashboard.add(Box.createRigidArea(new Dimension(0, 6)));

        JPanel statsPanel = new JPanel(new GridLayout(4, 2, 6, 4));
        statsPanel.setOpaque(false);
        statsPanel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(60, 65, 80)), "Training Metrics",
                TitledBorder.LEFT, TitledBorder.TOP, new Font("SansSerif", Font.BOLD, 11), Color.WHITE));

        lblEpisode = createStatLabel("Episode: 0");
        lblScore = createStatLabel("Score: 0");
        lblHighScore = createStatLabel("High Score: 0");
        lblReward = createStatLabel("Ep Reward: 0.0");
        lblAvgReward = createStatLabel("Avg (Last 20): 0.0");
        lblLoss = createStatLabel("Loss A/C: 0.00 / 0.00");
        lblFPS = createStatLabel("FPS: 60");
        lblActionName = createStatLabel("Action: NONE");

        statsPanel.add(lblEpisode);
        statsPanel.add(lblScore);
        statsPanel.add(lblHighScore);
        statsPanel.add(lblReward);
        statsPanel.add(lblAvgReward);
        statsPanel.add(lblLoss);
        statsPanel.add(lblFPS);
        statsPanel.add(lblActionName);

        dashboard.add(statsPanel);
        dashboard.add(Box.createRigidArea(new Dimension(0, 6)));

        JPanel actionPanel = new JPanel(new GridLayout(5, 2, 4, 2));
        actionPanel.setOpaque(false);
        actionPanel.setBorder(BorderFactory.createTitledBorder(
                BorderFactory.createLineBorder(new Color(60, 65, 80)), "Actor Policy Distribution",
                TitledBorder.LEFT, TitledBorder.TOP, new Font("SansSerif", Font.BOLD, 11), Color.WHITE));

        String[] actionLabels = { "0: Stand Still", "1: Move Left", "2: Move Right", "3: Jump", "4: Right + Jump" };
        barActionProbs = new JProgressBar[5];

        for (int i = 0; i < 5; i++) {
            JLabel lbl = new JLabel(actionLabels[i] + ":", SwingConstants.RIGHT);
            lbl.setForeground(Color.LIGHT_GRAY);
            lbl.setFont(new Font("SansSerif", Font.PLAIN, 10));

            barActionProbs[i] = new JProgressBar(0, 100);
            barActionProbs[i].setStringPainted(true);
            barActionProbs[i].setFont(new Font("Monospaced", Font.BOLD, 9));

            actionPanel.add(lbl);
            actionPanel.add(barActionProbs[i]);
        }

        dashboard.add(actionPanel);
        dashboard.add(Box.createRigidArea(new Dimension(0, 6)));

        graphPanel = new GraphPanel(rewardHistory, avgRewardHistory);
        graphPanel.setPreferredSize(new Dimension(DASHBOARD_WIDTH - 20, 110));
        dashboard.add(graphPanel);
        dashboard.add(Box.createRigidArea(new Dimension(0, 6)));

        // Interactive Buttons
        JPanel controlPanel = new JPanel(new GridLayout(3, 2, 4, 4));
        controlPanel.setOpaque(false);

        btnToggleMode = new JButton("Mode: AI Agent");
        btnToggleMode.addActionListener(e -> toggleHumanMode());

        btnToggleSpeed = new JButton("Speed: Real-Time");
        btnToggleSpeed.addActionListener(e -> toggleSpeed());

        btnPause = new JButton("Pause Training");
        btnPause.addActionListener(e -> {
            isPaused = !isPaused;
            btnPause.setText(isPaused ? "Resume Training" : "Pause Training");
        });

        btnReset = new JButton("Reset Weights");
        btnReset.addActionListener(e -> resetAgent());

        btnSave = new JButton("Save Model");
        btnSave.addActionListener(e -> saveModel());

        btnLoad = new JButton("Load Model");
        btnLoad.addActionListener(e -> loadModel());

        styleButton(btnToggleMode);
        styleButton(btnToggleSpeed);
        styleButton(btnPause);
        styleButton(btnReset);
        styleButton(btnSave);
        styleButton(btnLoad);

        controlPanel.add(btnToggleMode);
        controlPanel.add(btnToggleSpeed);
        controlPanel.add(btnPause);
        controlPanel.add(btnReset);
        controlPanel.add(btnSave);
        controlPanel.add(btnLoad);

        dashboard.add(controlPanel);

        mainContainer.add(dashboard, BorderLayout.EAST);
        add(mainContainer);
    }

    private JLabel createStatLabel(String text) {
        JLabel lbl = new JLabel(text);
        lbl.setForeground(new Color(220, 225, 235));
        lbl.setFont(new Font("Monospaced", Font.PLAIN, 10));
        return lbl;
    }

    private void styleButton(JButton btn) {
        btn.setBackground(new Color(40, 45, 60));
        btn.setForeground(Color.WHITE);
        btn.setFocusPainted(false);
        btn.setFont(new Font("SansSerif", Font.BOLD, 10));
        btn.setBorder(BorderFactory.createCompoundBorder(
                BorderFactory.createLineBorder(new Color(70, 75, 95)),
                BorderFactory.createEmptyBorder(3, 3, 3, 3)));
    }

    private void gameLoop() {
        if (isPaused)
            return;

        // Calculate Real-Time Frame Rate
        fpsCounter++;
        long now = System.currentTimeMillis();
        if (now - lastFrameTime >= 1000) {
            currentFPS = fpsCounter;
            fpsCounter = 0;
            lastFrameTime = now;
        }

        int stepsToRun = (isFastForward && !isHumanMode) ? fastForwardSpeed : 1;

        for (int step = 0; step < stepsToRun; step++) {
            double[] state = env.getStateVector();
            int action;
            double value = 0.0;
            double logProb = 0.0;
            double[] actionProbs = new double[] { 0.2, 0.2, 0.2, 0.2, 0.2 };

            if (isHumanMode) {
                action = env.getHumanAction();
            } else {
                // Query PPO Policy Network
                actionProbs = agent.getActorOutput(state);
                action = agent.sampleAction(actionProbs);
                value = agent.getCriticOutput(state);
                logProb = Math.log(actionProbs[action] + 1e-8);
            }

            // Step Environment Forward
            StepResult result = env.step(action);
            currentEpisodeReward += result.reward;

            // Visualizer Updates (First iteration of batch)
            if (step == 0) {
                for (int a = 0; a < 5; a++) {
                    barActionProbs[a].setValue((int) (actionProbs[a] * 100));
                    barActionProbs[a].setString(String.format("%.1f%%", actionProbs[a] * 100));
                }
                String[] actionNames = { "STILL", "LEFT", "RIGHT", "JUMP", "RIGHT+JUMP" };
                lblActionName.setText("Action: " + actionNames[action]);
            }

            // Record Experience in PPO Rollout Buffer
            if (!isHumanMode) {
                agent.storeExperience(state, action, result.reward, value, logProb, result.done);
            }

            // Handle Episode Termination
            if (result.done) {
                episodeCount++;
                if (env.getScore() > highScore)
                    highScore = env.getScore();

                rewardHistory.add(currentEpisodeReward);
                double sum = 0;
                int count = 0;
                for (int j = Math.max(0, rewardHistory.size() - 20); j < rewardHistory.size(); j++) {
                    sum += rewardHistory.get(j);
                    count++;
                }
                avgRewardHistory.add(sum / count);

                // Perform PPO Stochastic Gradient Descent Update
                if (!isHumanMode && agent.getBufferLength() >= 256) {
                    double[] losses = agent.train();
                    lastActorLoss = losses[0];
                    lastCriticLoss = losses[1];
                }

                currentEpisodeReward = 0.0;
                env.reset();
            }
        }

        lblEpisode.setText("Episode: " + episodeCount);
        lblScore.setText("Score: " + env.getScore());
        lblHighScore.setText("High Score: " + highScore);
        lblReward.setText(String.format("Ep Reward: %.1f", currentEpisodeReward));
        lblAvgReward.setText(String.format("Avg (20): %.1f",
                avgRewardHistory.isEmpty() ? 0.0 : avgRewardHistory.get(avgRewardHistory.size() - 1)));
        lblLoss.setText(String.format("Loss A/C: %.2f / %.2f", lastActorLoss, lastCriticLoss));
        lblFPS.setText("FPS: " + currentFPS);

        gamePanel.repaint();
        graphPanel.repaint();

        if (isHumanMode)
            gamePanel.requestFocusInWindow();
    }

    private void toggleHumanMode() {
        isHumanMode = !isHumanMode;
        if (isHumanMode) {
            isFastForward = false;
            btnToggleSpeed.setText("Speed: Real-Time");
            btnToggleMode.setText("Mode: Human Control");
            btnToggleMode.setBackground(new Color(180, 70, 50));
            gamePanel.requestFocusInWindow();
        } else {
            btnToggleMode.setText("Mode: AI Agent");
            btnToggleMode.setBackground(new Color(40, 45, 60));
        }
        env.reset();
    }

    private void toggleSpeed() {
        if (isHumanMode)
            return;
        isFastForward = !isFastForward;
        btnToggleSpeed.setText(isFastForward ? "Speed: Fast (25x)" : "Speed: Real-Time");
        btnToggleSpeed.setBackground(isFastForward ? new Color(0, 140, 90) : new Color(40, 45, 60));
    }

    private void resetAgent() {
        agent = new PPOAgent(10, 5);
        rewardHistory.clear();
        avgRewardHistory.clear();
        episodeCount = 0;
        highScore = 0;
        env.reset();
        graphPanel.repaint();
        JOptionPane.showMessageDialog(this, "PPO Policy/Value Neural Weights Reset Successfully.");
    }

    private void saveModel() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setSelectedFile(new File("ppo_mario_model.bin"));
        if (fileChooser.showSaveDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                agent.saveWeights(fileChooser.getSelectedFile().getAbsolutePath());
                JOptionPane.showMessageDialog(this, "Model saved successfully!");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Error saving model: " + ex.getMessage());
            }
        }
    }

    private void loadModel() {
        JFileChooser fileChooser = new JFileChooser();
        if (fileChooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
            try {
                agent.loadWeights(fileChooser.getSelectedFile().getAbsolutePath());
                JOptionPane.showMessageDialog(this, "Model weights loaded successfully!");
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this, "Error loading model: " + ex.getMessage());
            }
        }
    }

    /*
     * ========================================================================
     * NEURAL NETWORK & MATRIX MATH LIBRARY (CUSTOM IMPLEMENTATION FROM SCRATCH)
     * ========================================================================
     */

    public static class Matrix implements Serializable {
        private static final long serialVersionUID = 1L;
        public final int rows;
        public final int cols;
        public final double[][] data;

        public Matrix(int rows, int cols) {
            this.rows = rows;
            this.cols = cols;
            this.data = new double[rows][cols];
        }

        public Matrix(double[][] data) {
            this.rows = data.length;
            this.cols = data[0].length;
            this.data = new double[rows][cols];
            for (int i = 0; i < rows; i++) {
                System.arraycopy(data[i], 0, this.data[i], 0, cols);
            }
        }

        public static Matrix randomXavier(int rows, int cols) {
            Matrix m = new Matrix(rows, cols);
            double std = Math.sqrt(2.0 / (rows + cols));
            Random rand = new Random();
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    m.data[i][j] = rand.nextGaussian() * std;
                }
            }
            return m;
        }

        public static Matrix multiply(Matrix a, Matrix b) {
            if (a.cols != b.rows)
                throw new IllegalArgumentException("Dimension mismatch");
            Matrix res = new Matrix(a.rows, b.cols);
            for (int i = 0; i < a.rows; i++) {
                for (int k = 0; k < a.cols; k++) {
                    for (int j = 0; j < b.cols; j++) {
                        res.data[i][j] += a.data[i][k] * b.data[k][j];
                    }
                }
            }
            return res;
        }

        public Matrix addRowVector(Matrix v) {
            Matrix res = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] = this.data[i][j] + v.data[0][j];
                }
            }
            return res;
        }

        public Matrix elementwiseMultiply(Matrix b) {
            Matrix res = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] = this.data[i][j] * b.data[i][j];
                }
            }
            return res;
        }

        public Matrix transpose() {
            Matrix res = new Matrix(cols, rows);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[j][i] = this.data[i][j];
                }
            }
            return res;
        }

        public Matrix relu() {
            Matrix res = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] = Math.max(0.01 * this.data[i][j], this.data[i][j]); // Leaky ReLU
                }
            }
            return res;
        }

        public Matrix reluDerivative() {
            Matrix res = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] = this.data[i][j] > 0 ? 1.0 : 0.01;
                }
            }
            return res;
        }

        public Matrix softmax() {
            Matrix res = new Matrix(rows, cols);
            for (int i = 0; i < rows; i++) {
                double max = Double.NEGATIVE_INFINITY;
                for (int j = 0; j < cols; j++) {
                    if (this.data[i][j] > max)
                        max = this.data[i][j];
                }
                double sum = 0.0;
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] = Math.exp(this.data[i][j] - max);
                    sum += res.data[i][j];
                }
                for (int j = 0; j < cols; j++) {
                    res.data[i][j] /= (sum + 1e-8);
                }
            }
            return res;
        }

        public Matrix sumColumns() {
            Matrix res = new Matrix(1, cols);
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    res.data[0][j] += this.data[i][j];
                }
            }
            return res;
        }
    }

    public static class DenseLayer implements Serializable {
        private static final long serialVersionUID = 1L;
        public Matrix weights;
        public Matrix bias;

        // Adam Parameters
        private Matrix mW, vW, mB, vB;
        private int t = 0;

        public transient Matrix lastInput;
        public transient Matrix lastLinearOutput;
        public transient Matrix lastActivationOutput;

        public final String activation;

        public DenseLayer(int inSize, int outSize, String activation) {
            this.weights = Matrix.randomXavier(inSize, outSize);
            this.bias = new Matrix(1, outSize);
            this.activation = activation;

            mW = new Matrix(inSize, outSize);
            vW = new Matrix(inSize, outSize);
            mB = new Matrix(1, outSize);
            vB = new Matrix(1, outSize);
        }

        public Matrix forward(Matrix input) {
            this.lastInput = input;
            this.lastLinearOutput = Matrix.multiply(input, weights).addRowVector(bias);

            if ("relu".equalsIgnoreCase(activation)) {
                this.lastActivationOutput = this.lastLinearOutput.relu();
            } else if ("softmax".equalsIgnoreCase(activation)) {
                this.lastActivationOutput = this.lastLinearOutput.softmax();
            } else {
                this.lastActivationOutput = this.lastLinearOutput;
            }
            return this.lastActivationOutput;
        }

        public Matrix backward(Matrix dL_dOut) {
            Matrix dL_dZ;
            if ("relu".equalsIgnoreCase(activation)) {
                dL_dZ = dL_dOut.elementwiseMultiply(lastLinearOutput.reluDerivative());
            } else {
                dL_dZ = dL_dOut;
            }

            Matrix dL_dW = Matrix.multiply(lastInput.transpose(), dL_dZ);
            Matrix dL_dB = dL_dZ.sumColumns();
            Matrix dL_dIn = Matrix.multiply(dL_dZ, weights.transpose());

            updateAdam(dL_dW, dL_dB, 0.0005, 0.9, 0.999, 1e-8);
            return dL_dIn;
        }

        private void updateAdam(Matrix dW, Matrix dB, double lr, double beta1, double beta2, double eps) {
            t++;
            double b1Factor = 1.0 - Math.pow(beta1, t);
            double b2Factor = 1.0 - Math.pow(beta2, t);

            for (int i = 0; i < weights.rows; i++) {
                for (int j = 0; j < weights.cols; j++) {
                    double grad = Math.max(-5.0, Math.min(5.0, dW.data[i][j])); // Clip gradient
                    mW.data[i][j] = beta1 * mW.data[i][j] + (1 - beta1) * grad;
                    vW.data[i][j] = beta2 * vW.data[i][j] + (1 - beta2) * grad * grad;

                    double mHat = mW.data[i][j] / b1Factor;
                    double vHat = vW.data[i][j] / b2Factor;
                    weights.data[i][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
                }
            }

            for (int j = 0; j < bias.cols; j++) {
                double grad = Math.max(-5.0, Math.min(5.0, dB.data[0][j]));
                mB.data[0][j] = beta1 * mB.data[0][j] + (1 - beta1) * grad;
                vB.data[0][j] = beta2 * vB.data[0][j] + (1 - beta2) * grad * grad;

                double mHat = mB.data[0][j] / b1Factor;
                double vHat = vB.data[0][j] / b2Factor;
                bias.data[0][j] -= lr * mHat / (Math.sqrt(vHat) + eps);
            }
        }
    }

    public static class NeuralNetwork implements Serializable {
        private static final long serialVersionUID = 1L;
        public final List<DenseLayer> layers = new ArrayList<>();

        public NeuralNetwork(int inputDim, int hiddenDim, int outputDim, String finalActivation) {
            // Input -> 128 ReLU -> 128 ReLU -> Output
            layers.add(new DenseLayer(inputDim, hiddenDim, "relu"));
            layers.add(new DenseLayer(hiddenDim, hiddenDim, "relu"));
            layers.add(new DenseLayer(hiddenDim, outputDim, finalActivation));
        }

        public Matrix forward(Matrix input) {
            Matrix curr = input;
            for (DenseLayer layer : layers) {
                curr = layer.forward(curr);
            }
            return curr;
        }

        public void backpropagate(Matrix outputGrad) {
            Matrix currGrad = outputGrad;
            for (int i = layers.size() - 1; i >= 0; i--) {
                currGrad = layers.get(i).backward(currGrad);
            }
        }
    }

    /*
     * ========================================================================
     * PROXIMAL POLICY OPTIMIZATION (PPO) AGENT IMPLEMENTATION
     * ========================================================================
     */

    public static class Experience {
        double[] state;
        int action;
        double reward;
        double value;
        double logProb;
        boolean done;

        public Experience(double[] state, int action, double reward, double value, double logProb, boolean done) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.value = value;
            this.logProb = logProb;
            this.done = done;
        }
    }

    public static class PPOAgent implements Serializable {
        private static final long serialVersionUID = 1L;

        private NeuralNetwork actor;
        private NeuralNetwork critic;

        // PPO Hyperparameters
        private final double gamma = 0.99;
        private final double gaeLambda = 0.95;
        private final double clipEpsilon = 0.2;
        private final double entropyCoeff = 0.01;
        private final int ppoEpochs = 4;

        private final transient List<Experience> buffer = new ArrayList<>();
        private final Random random = new Random();

        public PPOAgent(int stateDim, int actionDim) {
            actor = new NeuralNetwork(stateDim, 128, actionDim, "softmax");
            critic = new NeuralNetwork(stateDim, 128, 1, "none");
        }

        public double[] getActorOutput(double[] state) {
            Matrix input = new Matrix(new double[][] { state });
            return actor.forward(input).data[0];
        }

        public double getCriticOutput(double[] state) {
            Matrix input = new Matrix(new double[][] { state });
            return critic.forward(input).data[0][0];
        }

        public int sampleAction(double[] probs) {
            double r = random.nextDouble();
            double cumulative = 0.0;
            for (int i = 0; i < probs.length; i++) {
                cumulative += probs[i];
                if (r <= cumulative)
                    return i;
            }
            return probs.length - 1;
        }

        public void storeExperience(double[] state, int action, double reward, double value, double logProb,
                boolean done) {
            buffer.add(new Experience(state, action, reward, value, logProb, done));
        }

        public int getBufferLength() {
            return buffer.size();
        }

        public double[] train() {
            int N = buffer.size();
            double[] advantages = new double[N];
            double[] returns = new double[N];

            // 1. Calculate Generalized Advantage Estimation (GAE)
            double gae = 0.0;
            for (int t = N - 1; t >= 0; t--) {
                Experience exp = buffer.get(t);
                double nextValue = (t == N - 1 || exp.done) ? 0.0 : buffer.get(t + 1).value;
                double delta = exp.reward + gamma * nextValue * (exp.done ? 0 : 1) - exp.value;
                gae = delta + gamma * gaeLambda * (exp.done ? 0 : 1) * gae;
                advantages[t] = gae;
                returns[t] = advantages[t] + exp.value;
            }

            // Advantage Normalization
            double advMean = 0.0;
            for (double a : advantages)
                advMean += a;
            advMean /= N;

            double advVar = 0.0;
            for (double a : advantages)
                advVar += (a - advMean) * (a - advMean);
            double advStd = Math.sqrt(advVar / N) + 1e-8;

            for (int i = 0; i < N; i++) {
                advantages[i] = (advantages[i] - advMean) / advStd;
            }

            // 2. Perform PPO Epoch Optimization
            double totalActorLoss = 0.0;
            double totalCriticLoss = 0.0;

            for (int epoch = 0; epoch < ppoEpochs; epoch++) {
                for (int i = 0; i < N; i++) {
                    Experience exp = buffer.get(i);

                    // Actor Network Backpropagation
                    Matrix stateMat = new Matrix(new double[][] { exp.state });
                    Matrix probs = actor.forward(stateMat);
                    double currentProb = probs.data[0][exp.action];
                    double ratio = Math.exp(Math.log(currentProb + 1e-8) - exp.logProb);

                    double adv = advantages[i];
                    double surr1 = ratio * adv;
                    double surr2 = Math.max(1.0 - clipEpsilon, Math.min(1.0 + clipEpsilon, ratio)) * adv;

                    Matrix actorGrad = new Matrix(1, probs.cols);
                    for (int a = 0; a < probs.cols; a++) {
                        double p = probs.data[0][a];
                        double dEntropy = -(Math.log(p + 1e-8) + 1.0);

                        if (a == exp.action) {
                            double gradPolicy = (surr1 <= surr2 ? ratio
                                    : (adv >= 0 ? 1.0 + clipEpsilon : 1.0 - clipEpsilon)) * adv;
                            actorGrad.data[0][a] = -gradPolicy - entropyCoeff * dEntropy;
                        } else {
                            actorGrad.data[0][a] = -entropyCoeff * dEntropy;
                        }
                    }

                    actor.backpropagate(actorGrad);
                    totalActorLoss += -Math.min(surr1, surr2);

                    // Critic Network MSE Backpropagation
                    Matrix currentVal = critic.forward(stateMat);
                    double valError = currentVal.data[0][0] - returns[i];

                    Matrix criticGrad = new Matrix(1, 1);
                    criticGrad.data[0][0] = valError;

                    critic.backpropagate(criticGrad);
                    totalCriticLoss += valError * valError;
                }
            }

            buffer.clear();
            return new double[] { totalActorLoss / (N * ppoEpochs), totalCriticLoss / (N * ppoEpochs) };
        }

        public void saveWeights(String filename) throws IOException {
            try (ObjectOutputStream oos = new ObjectOutputStream(new FileOutputStream(filename))) {
                oos.writeObject(actor);
                oos.writeObject(critic);
            }
        }

        public void loadWeights(String filename) throws IOException, ClassNotFoundException {
            try (ObjectInputStream ois = new ObjectInputStream(new FileInputStream(filename))) {
                actor = (NeuralNetwork) ois.readObject();
                critic = (NeuralNetwork) ois.readObject();
            }
        }
    }

    /*
     * ========================================================================
     * 2D MARIO-STYLE PLATFORMER GAME ENVIRONMENT & PHYSICS ENGINE
     * ========================================================================
     */

    public static class StepResult {
        public double[] state;
        public double reward;
        public boolean done;

        public StepResult(double[] state, double reward, boolean done) {
            this.state = state;
            this.reward = reward;
            this.done = done;
        }
    }

    public static class Platform {
        public double x, y, width, height;

        public Platform(double x, double y, double width, double height) {
            this.x = x;
            this.y = y;
            this.width = width;
            this.height = height;
        }
    }

    public static class Enemy {
        public double x, y, width = 24, height = 24;
        public double vx = -1.5;
        public boolean alive = true;

        public Enemy(double x, double y) {
            this.x = x;
            this.y = y;
        }

        public void update(List<Platform> platforms) {
            if (!alive)
                return;
            x += vx;

            // Reverse direction on platform edges
            boolean onGround = false;
            for (Platform p : platforms) {
                if (x + width >= p.x && x <= p.x + p.width && Math.abs((y + height) - p.y) < 2) {
                    onGround = true;
                    if (x <= p.x || x + width >= p.x + p.width) {
                        vx = -vx;
                    }
                    break;
                }
            }
            if (!onGround)
                vx = -vx;
        }
    }

    public static class Coin {
        public double x, y, width = 16, height = 16;
        public boolean collected = false;

        public Coin(double x, double y) {
            this.x = x;
            this.y = y;
        }
    }

    public static class MarioEnv {
        private final int levelWidth;
        private final int levelHeight;

        private double playerX, playerY;
        private double playerVx, playerVy;
        private final double gravity = 0.5;
        private final double moveSpeed = 4.0;
        private final double jumpForce = -10.5;
        private boolean isGrounded = false;

        private List<Platform> platforms;
        private List<Enemy> enemies;
        private List<Coin> coins;
        private double goalX;

        private int score;
        private boolean gameOver;
        private boolean gameWon;
        private double prevPlayerX;
        private int humanAction = 0;

        public MarioEnv(int levelWidth, int levelHeight) {
            this.levelWidth = levelWidth;
            this.levelHeight = levelHeight;
            reset();
        }

        public void reset() {
            playerX = 60;
            playerY = 280;
            playerVx = 0;
            playerVy = 0;
            prevPlayerX = playerX;
            isGrounded = false;
            score = 0;
            gameOver = false;
            gameWon = false;
            humanAction = 0;

            generateLevelLayout();
        }

        private void generateLevelLayout() {
            platforms = new ArrayList<>();
            enemies = new ArrayList<>();
            coins = new ArrayList<>();

            // Ground Platforms with Gaps (Pitfalls)
            platforms.add(new Platform(0, 360, 600, 90));
            platforms.add(new Platform(680, 360, 500, 90));
            platforms.add(new Platform(1260, 360, 600, 90));
            platforms.add(new Platform(1920, 360, 680, 90));

            // Floating Platform Stairs and Bridges
            platforms.add(new Platform(200, 260, 120, 20));
            platforms.add(new Platform(400, 200, 100, 20));

            platforms.add(new Platform(750, 260, 140, 20));
            platforms.add(new Platform(950, 190, 120, 20));

            platforms.add(new Platform(1350, 270, 100, 20));
            platforms.add(new Platform(1500, 200, 120, 20));
            platforms.add(new Platform(1680, 270, 100, 20));

            platforms.add(new Platform(2050, 260, 140, 20));

            // Enemies
            enemies.add(new Enemy(350, 336));
            enemies.add(new Enemy(820, 336));
            enemies.add(new Enemy(1450, 336));
            enemies.add(new Enemy(1530, 176));
            enemies.add(new Enemy(2100, 336));

            // Coins
            coins.add(new Coin(230, 220));
            coins.add(new Coin(260, 220));
            coins.add(new Coin(430, 160));
            coins.add(new Coin(780, 220));
            coins.add(new Coin(990, 150));
            coins.add(new Coin(1530, 160));
            coins.add(new Coin(2080, 220));

            // Goal Flag Pole Position
            goalX = 2450;
        }

        public void setHumanAction(int action) {
            this.humanAction = action;
        }

        public int getHumanAction() {
            return humanAction;
        }

        public StepResult step(int action) {
            if (gameOver || gameWon)
                return new StepResult(getStateVector(), 0.0, true);

            // Apply Actions
            // 0: Stand, 1: Left, 2: Right, 3: Jump, 4: Right + Jump
            if (action == 1)
                playerVx = -moveSpeed;
            else if (action == 2)
                playerVx = moveSpeed;
            else if (action == 3 && isGrounded) {
                playerVy = jumpForce;
                isGrounded = false;
            } else if (action == 4) {
                playerVx = moveSpeed;
                if (isGrounded) {
                    playerVy = jumpForce;
                    isGrounded = false;
                }
            } else {
                playerVx *= 0.7; // Friction
            }

            // Gravity update
            playerVy += gravity;

            // Apply Horizontal Movement & Platform Side Collision
            playerX += playerVx;
            if (playerX < 0)
                playerX = 0;

            // Apply Vertical Movement & Platform Top/Bottom Collision
            playerY += playerVy;
            isGrounded = false;

            for (Platform p : platforms) {
                // Check AABB Overlap
                if (playerX + 24 > p.x && playerX < p.x + p.width) {
                    // Standing on top of platform
                    if (playerY + 28 >= p.y && playerY + 28 <= p.y + p.height && playerVy >= 0) {
                        playerY = p.y - 28;
                        playerVy = 0;
                        isGrounded = true;
                    }
                }
            }

            // Update Enemies
            for (Enemy e : enemies)
                e.update(platforms);

            double reward = -0.05; // Time waste penalty

            // Reward for Rightward Movement Progress
            double deltaX = playerX - prevPlayerX;
            if (deltaX > 0)
                reward += deltaX * 0.05;
            prevPlayerX = playerX;

            // Collect Coins (+10 Reward)
            for (Coin c : coins) {
                if (!c.collected && Math.abs(playerX - c.x) < 20 && Math.abs(playerY - c.y) < 20) {
                    c.collected = true;
                    score += 10;
                    reward += 10.0;
                }
            }

            // Enemy Collision Check (+20 Stomp Reward / -100 Death Penalty)
            for (Enemy e : enemies) {
                if (e.alive && playerX + 24 > e.x && playerX < e.x + e.width && playerY + 28 > e.y
                        && playerY < e.y + e.height) {
                    if (playerVy > 0 && playerY + 28 < e.y + 12) { // Stomp on enemy head
                        e.alive = false;
                        score += 20;
                        reward += 20.0;
                        playerVy = jumpForce * 0.6; // Bounce impulse
                    } else { // Player hit by enemy
                        gameOver = true;
                        reward = -100.0;
                    }
                }
            }

            // Pitfall Death Collision (-100 Penalty)
            if (playerY > levelHeight) {
                gameOver = true;
                reward = -100.0;
            }

            // Goal Flag Reached (+50 Win Reward)
            if (playerX >= goalX) {
                gameWon = true;
                score += 50;
                reward = 50.0;
            }

            boolean done = gameOver || gameWon;
            return new StepResult(getStateVector(), reward, done);
        }

        public double[] getStateVector() {
            Platform nearestPlat = getNearestPlatform();
            Enemy nearestEnemy = getNearestEnemy();

            double platDistX = nearestPlat != null ? (nearestPlat.x - playerX) / (double) levelWidth : 1.0;
            double platDistY = nearestPlat != null ? (nearestPlat.y - playerY) / (double) levelHeight : 1.0;

            double enemyDist = nearestEnemy != null
                    ? Math.hypot(nearestEnemy.x - playerX, nearestEnemy.y - playerY) / (double) levelWidth
                    : 1.0;
            double enemyDirX = nearestEnemy != null ? Math.signum(nearestEnemy.x - playerX) : 0.0;

            return new double[] {
                    playerX / (double) levelWidth, // 1. Player X normalized
                    playerY / (double) levelHeight, // 2. Player Y normalized
                    playerVx / moveSpeed, // 3. Horizontal Velocity
                    playerVy / 15.0, // 4. Vertical Velocity
                    platDistX, // 5. Relative Platform X Distance
                    platDistY, // 6. Relative Platform Y Distance
                    enemyDist, // 7. Relative Enemy Distance
                    enemyDirX, // 8. Enemy Direction (-1 to +1)
                    (goalX - playerX) / (double) levelWidth, // 9. Goal Distance
                    isGrounded ? 1.0 : 0.0 // 10. Grounded status flag
            };
        }

        private Platform getNearestPlatform() {
            Platform nearest = null;
            double minDist = Double.MAX_VALUE;
            for (Platform p : platforms) {
                if (p.x + p.width >= playerX) {
                    double d = p.x - playerX;
                    if (d < minDist) {
                        minDist = d;
                        nearest = p;
                    }
                }
            }
            return nearest;
        }

        private Enemy getNearestEnemy() {
            Enemy nearest = null;
            double minDist = Double.MAX_VALUE;
            for (Enemy e : enemies) {
                if (e.alive) {
                    double d = Math.abs(e.x - playerX);
                    if (d < minDist) {
                        minDist = d;
                        nearest = e;
                    }
                }
            }
            return nearest;
        }

        public double getPlayerX() {
            return playerX;
        }

        public double getPlayerY() {
            return playerY;
        }

        public List<Platform> getPlatforms() {
            return platforms;
        }

        public List<Enemy> getEnemies() {
            return enemies;
        }

        public List<Coin> getCoins() {
            return coins;
        }

        public double getGoalX() {
            return goalX;
        }

        public int getScore() {
            return score;
        }

        public boolean isGameOver() {
            return gameOver;
        }

        public boolean isGameWon() {
            return gameWon;
        }
    }

    /*
     * ========================================================================
     * GUI GRAPHICS & RENDERING PANELS
     * ========================================================================
     */

    public static class GameCanvas extends JPanel {
        private final MarioEnv env;

        public GameCanvas(MarioEnv env) {
            this.env = env;
            setBackground(new Color(100, 150, 240)); // Classic Sky Blue
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            // Dynamic Camera Follow Offset
            double cameraX = Math.max(0, Math.min(env.getPlayerX() - 150, PPOMarioGame.LEVEL_WIDTH - getWidth()));

            g2.translate(-cameraX, 0);

            // 1. Draw Clouds & Hill Background Shapes
            g2.setColor(new Color(255, 255, 255, 180));
            g2.fillOval(200, 80, 100, 40);
            g2.fillOval(700, 60, 120, 50);
            g2.fillOval(1400, 90, 110, 40);
            g2.fillOval(2000, 70, 130, 45);

            // 2. Draw Platforms & Ground
            for (Platform p : env.getPlatforms()) {
                // Grass Top Layer
                g2.setColor(new Color(40, 180, 70));
                g2.fillRect((int) p.x, (int) p.y, (int) p.width, 6);

                // Dirt Block Base
                g2.setColor(new Color(150, 85, 40));
                g2.fillRect((int) p.x, (int) p.y + 6, (int) p.width, (int) p.height - 6);
                g2.setColor(new Color(100, 50, 20));
                g2.drawRect((int) p.x, (int) p.y, (int) p.width, (int) p.height);
            }

            // 3. Draw Collectible Coins
            for (Coin c : env.getCoins()) {
                if (!c.collected) {
                    g2.setColor(new Color(255, 215, 0));
                    g2.fillOval((int) c.x, (int) c.y, (int) c.width, (int) c.height);
                    g2.setColor(new Color(210, 160, 0));
                    g2.drawOval((int) c.x, (int) c.y, (int) c.width, (int) c.height);
                }
            }

            // 4. Draw Enemies (Goombas)
            for (Enemy e : env.getEnemies()) {
                if (e.alive) {
                    g2.setColor(new Color(180, 50, 30));
                    g2.fillRect((int) e.x, (int) e.y, (int) e.width, (int) e.height);
                    // Eyes
                    g2.setColor(Color.WHITE);
                    g2.fillRect((int) e.x + 4, (int) e.y + 4, 5, 5);
                    g2.fillRect((int) e.x + 15, (int) e.y + 4, 5, 5);
                    g2.setColor(Color.BLACK);
                    g2.fillRect((int) e.x + 6, (int) e.y + 6, 2, 2);
                    g2.fillRect((int) e.x + 17, (int) e.y + 6, 2, 2);
                }
            }

            // 5. Draw Goal Flag Pole
            int gX = (int) env.getGoalX();
            g2.setColor(Color.LIGHT_GRAY);
            g2.fillRect(gX, 160, 8, 200);
            g2.setColor(new Color(230, 40, 40));
            g2.fillPolygon(new int[] { gX + 8, gX + 50, gX + 8 }, new int[] { 160, 180, 200 }, 3);

            // 6. Draw Player Character (Mario Sprite Box)
            int pX = (int) env.getPlayerX();
            int pY = (int) env.getPlayerY();

            // Red Shirt / Cap
            g2.setColor(new Color(230, 30, 30));
            g2.fillRect(pX + 2, pY, 20, 8);
            g2.fillRect(pX + 4, pY + 8, 16, 12);

            // Blue Overalls
            g2.setColor(new Color(30, 80, 200));
            g2.fillRect(pX + 4, pY + 16, 16, 12);

            // Eyes & Cap Rim
            g2.setColor(Color.WHITE);
            g2.fillRect(pX + 14, pY + 4, 4, 4);

            g2.translate(cameraX, 0); // Reset camera transform

            // Overlay Banners
            if (env.isGameOver()) {
                g2.setColor(new Color(0, 0, 0, 150));
                g2.fillRect(0, 0, getWidth(), getHeight());
                g2.setColor(Color.RED);
                g2.setFont(new Font("SansSerif", Font.BOLD, 30));
                g2.drawString("GAME OVER", getWidth() / 2 - 90, getHeight() / 2);
            } else if (env.isGameWon()) {
                g2.setColor(new Color(0, 0, 0, 130));
                g2.fillRect(0, 0, getWidth(), getHeight());
                g2.setColor(Color.GREEN);
                g2.setFont(new Font("SansSerif", Font.BOLD, 30));
                g2.drawString("STAGE CLEARED!", getWidth() / 2 - 130, getHeight() / 2);
            }
        }
    }

    public static class GraphPanel extends JPanel {
        private final List<Double> rewards;
        private final List<Double> avgRewards;

        public GraphPanel(List<Double> rewards, List<Double> avgRewards) {
            this.rewards = rewards;
            this.avgRewards = avgRewards;
            setBackground(new Color(14, 16, 22));
            setBorder(BorderFactory.createTitledBorder(
                    BorderFactory.createLineBorder(new Color(60, 65, 80)), "Reward History",
                    TitledBorder.LEFT, TitledBorder.TOP, new Font("SansSerif", Font.BOLD, 11), Color.WHITE));
        }

        @Override
        protected void paintComponent(Graphics g) {
            super.paintComponent(g);
            if (rewards.size() < 2)
                return;

            Graphics2D g2 = (Graphics2D) g;
            g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);

            int w = getWidth() - 24;
            int h = getHeight() - 30;
            int offsetX = 16;
            int offsetY = 20;

            double maxR = 100.0;
            double minR = -100.0;
            for (double r : rewards) {
                if (r > maxR)
                    maxR = r;
                if (r < minR)
                    minR = r;
            }

            g2.setColor(new Color(60, 65, 80));
            g2.drawLine(offsetX, offsetY, offsetX, offsetY + h);
            g2.drawLine(offsetX, offsetY + h, offsetX + w, offsetY + h);

            int size = rewards.size();
            int startIndex = Math.max(0, size - 80);
            int count = size - startIndex;

            // Raw Episode Rewards (Light Cyan)
            g2.setColor(new Color(0, 200, 255, 100));
            for (int i = 0; i < count - 1; i++) {
                int idx1 = startIndex + i;
                int idx2 = startIndex + i + 1;

                int x1 = offsetX + (int) ((double) i / (count - 1) * w);
                int y1 = offsetY + h - (int) ((rewards.get(idx1) - minR) / (maxR - minR + 1e-5) * h);
                int x2 = offsetX + (int) ((double) (i + 1) / (count - 1) * w);
                int y2 = offsetY + h - (int) ((rewards.get(idx2) - minR) / (maxR - minR + 1e-5) * h);

                g2.drawLine(x1, y1, x2, y2);
            }

            // Moving Average Rewards (Orange Curve)
            g2.setColor(new Color(255, 140, 0));
            g2.setStroke(new BasicStroke(2.0f));
            for (int i = 0; i < count - 1; i++) {
                int idx1 = startIndex + i;
                int idx2 = startIndex + i + 1;

                int x1 = offsetX + (int) ((double) i / (count - 1) * w);
                int y1 = offsetY + h - (int) ((avgRewards.get(idx1) - minR) / (maxR - minR + 1e-5) * h);
                int x2 = offsetX + (int) ((double) (i + 1) / (count - 1) * w);
                int y2 = offsetY + h - (int) ((avgRewards.get(idx2) - minR) / (maxR - minR + 1e-5) * h);

                g2.drawLine(x1, y1, x2, y2);
            }
        }
    }

    public static void main(String[] args) {
        SwingUtilities.invokeLater(PPOMarioGame::new);
    }
}
