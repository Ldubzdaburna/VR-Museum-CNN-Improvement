using UnityEngine;
using Unity.Barracuda;
using System.Collections.Generic;
using System.Linq;
using System.Collections;
using TMPro;

public class Classification : MonoBehaviour
{
    const int IMAGE_SIZE = 244; // keep whatever the original project used
    const string INPUT_NAME = "images";
    const string OUTPUT_NAME = "Softmax";

    public NNModel modelFile;
    public TextAsset labelAsset;
    public CameraView CameraView;
    public Preprocess preprocess;
    public TMP_Text resText;

    public float confidenceThreshold = 0.7f;

    string[] labels;
    IWorker worker;

    void Start()
    {
        var model = ModelLoader.Load(modelFile);
        worker = WorkerFactory.CreateWorker(model, WorkerFactory.Device.GPU);

        var stringArray = labelAsset.text.Split('"').Where((item, index) => index % 2 != 0);
        labels = stringArray.Where((x, i) => i % 2 != 0).ToArray();
    }

    public void RunClass()
    {
        RenderTexture renderTexture = CameraView.GetCameraImage();
        if (renderTexture != null && renderTexture.width > 100)
        {
            preprocess.ScaleAndCropImage(renderTexture, IMAGE_SIZE, RunModel);
        }
    }

    void RunModel(byte[] pixels)
    {
        StopAllCoroutines();
        StartCoroutine(RunModelRoutine(pixels));
    }

    IEnumerator RunModelRoutine(byte[] pixels)
    {
        Tensor inputTensor = TransformInput(pixels);

        var inputs = new Dictionary<string, Tensor> { { INPUT_NAME, inputTensor } };
        worker.Execute(inputs);

        Tensor outputTensor = worker.CopyOutput(OUTPUT_NAME);
        var probs = outputTensor.ToReadOnlyArray();

        // Top-3 predictions
        var top3 = probs
            .Select((p, i) => new { p, i })
            .OrderByDescending(x => x.p)
            .Take(3)
            .ToList();

        float maxP = top3[0].p;
        string bestLabel = labels[top3[0].i];

        // Build result text (always show top-3)
        string resultText =
            $"{labels[top3[0].i]} ({top3[0].p * 100f:F1}%)\n" +
            $"{labels[top3[1].i]} ({top3[1].p * 100f:F1}%)\n" +
            $"{labels[top3[2].i]} ({top3[2].p * 100f:F1}%)";

        // If below threshold, prepend "Uncertain"
        if (maxP < confidenceThreshold)
        {
            resultText = $"Uncertain ({maxP * 100f:F1}%)\n" + resultText;
        }

        resText.text = resultText;

        inputTensor.Dispose();
        outputTensor.Dispose();
        yield return null;
    }

    Tensor TransformInput(byte[] pixels)
    {
        float[] transformedPixels = new float[pixels.Length];
        for (int i = 0; i < pixels.Length; i++)
        {
            transformedPixels[i] = (pixels[i] - 127f) / 128f;
        }
        return new Tensor(1, IMAGE_SIZE, IMAGE_SIZE, 3, transformedPixels);
    }

    void OnDestroy()
    {
        worker?.Dispose();
    }
}