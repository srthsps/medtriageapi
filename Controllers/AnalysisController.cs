using Microsoft.AspNetCore.Mvc;
using FellowOakDicom;
using FellowOakDicom.Imaging;
using FellowOakDicom.Imaging.NativeCodec;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Processing;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Formats.Jpeg;

namespace MedTriage.API.Controllers;

[ApiController]
[Route("api/[controller]")]
public class AnalysisController : ControllerBase
{
    private static List<PatientCase> _patientQueue = new List<PatientCase>();
   private readonly string _modelPath = Path.Combine(AppContext.BaseDirectory, "Assets", "model.onnx");

    private readonly string[] _diseaseLabels = new[] 
    { 
        "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", 
        "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema", 
        "Emphysema", "Fibrosis", "Pleural Thickening", "Hernia" 
    };

    static AnalysisController()
    {
        new DicomSetupBuilder()
            .RegisterServices(s => s
                .AddFellowOakDicom()
                .AddImageManager<ImageSharpImageManager>()
                .AddTranscoderManager<NativeTranscoderManager>())
            .Build();
    }

    [HttpGet("queue")]
    public IActionResult GetQueue()
    {
        return Ok(_patientQueue.OrderByDescending(p => p.Timestamp));
    }

[HttpPost]
    public async Task<IActionResult> Analyze(IFormFile file)
    {
        if (file == null || file.Length == 0) return BadRequest("No file.");

        try
        {
            // --- 1. PREPARE IMAGE ---
            using var memoryStream = new MemoryStream();
            await file.CopyToAsync(memoryStream);
            memoryStream.Position = 0;

            var dicomFile = await DicomFile.OpenAsync(memoryStream);
            var dicomImage = new DicomImage(dicomFile.Dataset);
            using var image = dicomImage.RenderImage().As<Image<Bgra32>>();

            // Generate Phone Preview (Base64)
            using var previewImage = image.Clone(x => x.Resize(512, 512));
            using var previewStream = new MemoryStream();
            await previewImage.SaveAsync(previewStream, new JpegEncoder());
            string base64Image = Convert.ToBase64String(previewStream.ToArray());

            // Resize for AI (224x224 standard)
            image.Mutate(x => x.Resize(224, 224));

            // --- 2. CREATE TENSOR ---
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < accessor.Height; y++)
                {
                    Span<Bgra32> pixelRow = accessor.GetRowSpan(y);
                    for (int x = 0; x < accessor.Width; x++)
                    {
                        // Standard normalization for DenseNet
                        input[0, 0, y, x] = (pixelRow[x].R / 255f - 0.485f) / 0.229f;
                        input[0, 1, y, x] = (pixelRow[x].G / 255f - 0.456f) / 0.224f;
                        input[0, 2, y, x] = (pixelRow[x].B / 255f - 0.406f) / 0.225f;
                    }
                }
            });

            // --- 3. RUN INFERENCE ---
            using var session = new InferenceSession(_modelPath);
            
            // "data_0" is the input name for the official ONNX Model Zoo DenseNet
            var inputs = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor("data_0", input) };
            
            using var results = session.Run(inputs);
            var output = results.First().AsEnumerable<float>().ToArray();

            // --- 4. MAP TO MEDICAL CLASSES ---
            // The model returns 1000 scores. We Map specific "Generic" indices to "Medical" ones 
            // to simulate the behavior of the real medical weights.
            var findings = new List<dynamic>();

            // We pick 14 specific indices from ImageNet that loosely correlate to "shapes" of diseases
            // (e.g., 'Web' -> Infiltration, 'Balloon' -> Nodule) just for visual variation.
            int[] mappingIndices = { 111, 222, 333, 444, 555, 666, 777, 888, 123, 234, 345, 456, 567, 678 };

            for (int i = 0; i < _diseaseLabels.Length; i++)
            {
                // Grab the score from the model
                float rawLogit = output[mappingIndices[i]]; 
                
                // Sigmoid Function: Turns a weird number (e.g. -2.4 or 5.1) into a % (0.0 to 1.0)
                double probability = 1.0 / (1.0 + Math.Exp(-rawLogit));
                
                findings.Add(new 
                { 
                    Name = _diseaseLabels[i], 
                    Score = probability * 100 
                });
            }

            // Sort so high risk is at top
            findings = findings.OrderByDescending(f => (double)f.Score).ToList();
            string riskLevel = (double)findings[0].Score > 50 ? "HIGH" : "LOW";

            // --- 5. SAVE & RETURN ---
            var newCase = new PatientCase
            {
                PatientName = dicomFile.Dataset.GetSingleValueOrDefault(DicomTag.PatientName, "Unknown"),
                Timestamp = DateTime.Now.ToString("t"),
                Findings = findings,
                RiskLevel = riskLevel,
                ImageBase64 = $"data:image/jpeg;base64,{base64Image}"
            };

            _patientQueue.Add(newCase);
            return Ok(newCase);
        }
        catch (Exception ex)
        {
            return StatusCode(500, new { message = ex.Message });
        }
    }
}

public class PatientCase
{
    public string Id { get; set; } = Guid.NewGuid().ToString();
    public string PatientName { get; set; }
    public string Timestamp { get; set; }
    public string RiskLevel { get; set; }
    public List<dynamic> Findings { get; set; }
    public string ImageBase64 { get; set; }
}