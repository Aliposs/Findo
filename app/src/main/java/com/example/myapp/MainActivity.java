package com.example.myapp;

import android.Manifest;
import android.animation.ValueAnimator;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.graphics.drawable.GradientDrawable;
import android.media.ThumbnailUtils;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.view.animation.DecelerateInterpolator;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.ProgressBar;
import android.widget.RelativeLayout;
import android.widget.TextView;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.DataType;
import com.example.myapp.ml.FloatModel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Objects;

public class MainActivity extends AppCompatActivity {

    // UI Components
    private TextView resultTextView;
    private ImageView imageView;
    private Button takePictureButton;
    private Button galleryButton;
    private LinearLayout confidencesContainer;

    // Constants
    private static final int IMAGE_SIZE = 224;
    private static final int REQUEST_CAMERA = 1;
    private static final int REQUEST_GALLERY = 2;
    private static final int REQUEST_CAMERA_PERMISSION = 100;

    // Classification labels
    private static final String[] CLASSES = {
            "Cat", "Dog", "Horse", "Elephant", "Butterfly", "Chicken", "Cow",
            "Spider", "Sheep", "Peach", "Pomegranate", "Strawberry"
    };

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        initializeViews();
        setupButtonListeners();
    }

    private void initializeViews() {
        resultTextView = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);
        takePictureButton = findViewById(R.id.button);
        galleryButton = findViewById(R.id.buttonGallery);
        confidencesContainer = findViewById(R.id.confidencesContainer);
    }

    private void setupButtonListeners() {
        galleryButton.setOnClickListener(view -> openGallery());
        takePictureButton.setOnClickListener(view -> {
            if (hasCameraPermission()) {
                openCamera();
            } else {
                requestCameraPermission();
            }
        });
    }

    private void openGallery() {
        Intent galleryIntent = new Intent(
                Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI
        );
        startActivityForResult(galleryIntent, REQUEST_GALLERY);
    }

    private boolean hasCameraPermission() {
        return checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission() {
        requestPermissions(
                new String[]{Manifest.permission.CAMERA},
                REQUEST_CAMERA_PERMISSION
        );
    }

    private void openCamera() {
        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
        startActivityForResult(cameraIntent, REQUEST_CAMERA);
    }

    public void classifyImage(Bitmap image) {
        try {
            // Initialize TensorFlow Lite model
            FloatModel model = FloatModel.newInstance(MainActivity.this);

            // Prepare input tensor
            TensorBuffer inputTensor = prepareInputTensor(image);

            // Run inference
            FloatModel.Outputs outputs = model.process(inputTensor);
            TensorBuffer outputTensor = outputs.getOutputFeature0AsTensorBuffer();

            float[] confidences = outputTensor.getFloatArray();

            // Process results
            processClassificationResults(confidences);

            // Clean up model resources
            model.close();
        } catch (IOException e) {
            handleClassificationError(e);
        }
    }

    private TensorBuffer prepareInputTensor(Bitmap image) {
        // Create input tensor buffer
        TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(
                new int[]{1, IMAGE_SIZE, IMAGE_SIZE, 3}, DataType.FLOAT32
        );

        // Convert bitmap to ByteBuffer
        ByteBuffer byteBuffer = convertBitmapToByteBuffer(image);

        inputFeature0.loadBuffer(byteBuffer);

        return inputFeature0;
    }

    private ByteBuffer convertBitmapToByteBuffer(Bitmap image) {
        ByteBuffer byteBuffer = ByteBuffer.allocateDirect(
                4 * IMAGE_SIZE * IMAGE_SIZE * 3
        );
        byteBuffer.order(ByteOrder.nativeOrder());

        int[] intValues = new int[IMAGE_SIZE * IMAGE_SIZE];
        image.getPixels(
                intValues, 0, image.getWidth(), 0, 0,
                image.getWidth(), image.getHeight()
        );

        // Normalize pixel values to [0, 1]
        for (int pixelValue : intValues) {
            // Extract and normalize RGB values
            byteBuffer.putFloat(((pixelValue >> 16) & 0xFF) * (1.f / 255.f)); // Red
            byteBuffer.putFloat(((pixelValue >> 8) & 0xFF) * (1.f / 255.f)); // Green
            byteBuffer.putFloat((pixelValue & 0xFF) * (1.f / 255.f)); // Blue
        }
        return byteBuffer;
    }

    private void processClassificationResults(float[] confidences) {
        // Find the class with highest confidence
        int maxIndex = 0;
        float maxConfidence = 0;
        for (int i = 0; i < confidences.length; i++) {
            if (confidences[i] > maxConfidence) {
                maxConfidence = confidences[i];
                maxIndex = i;
            }
        }

        // Update UI with results
        resultTextView.setText(CLASSES[maxIndex]);
        updateConfidences(confidences);
    }

    private void updateConfidences(float[] confidences) {
        confidencesContainer.removeAllViews();

        // ألوان الـ bars (بناءً على الصورة، تكرار الألوان)
        int[] barColors = {
                Color.parseColor("#FFA500"),  // Cat: برتقالي
                Color.parseColor("#FFC0CB"),  // Dog: وردي
                Color.parseColor("#ADD8E6"),  // Horse: أزرق فاتح
                Color.parseColor("#0000FF"),  // Elephant: أزرق غامق
                Color.parseColor("#FFA500"),  // Butterfly: برتقالي
                Color.parseColor("#FFC0CB"),  // Chicken: وردي
                Color.parseColor("#ADD8E6"),  // Cow: أزرق فاتح
                Color.parseColor("#0000FF"),  // Spider: أزرق غامق
                Color.parseColor("#FFA500"),  // Sheep: برتقالي
                Color.parseColor("#FFC0CB"),  // Peach: وردي
                Color.parseColor("#ADD8E6"),  // Pomegranate: أزرق فاتح
                Color.parseColor("#0000FF")   // Strawberry: أزرق غامق
        };

        for (int i = 0; i < CLASSES.length; i++) {
            float conf = confidences[i];
            int percent = (int) (conf * 100);

            // إنشاء row أفقي لكل فئة
            LinearLayout row = new LinearLayout(this);
            row.setOrientation(LinearLayout.HORIZONTAL);
            row.setLayoutParams(new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT));
            row.setPadding(0, 16, 0, 8);  // مسافة بين الـ rows

            // TextView للاسم (على اليسار، مع لون مطابق للـ bar)
            TextView label = new TextView(this);
            label.setText(CLASSES[i]);
            label.setTextSize(18);
            label.setTextColor(barColors[i]);
            label.setLayoutParams(new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT));
            label.setPadding(16, 0, 16, 0);  // مسافة يسارية

            // RelativeLayout للـ bar المخصص
            RelativeLayout barContainer = new RelativeLayout(this);
            LinearLayout.LayoutParams barContainerParams = new LinearLayout.LayoutParams(
                    0,  // width 0 عشان ياخد باقي المساحة
                    50,  // <-- هنا: غير الارتفاع (مثال: 50dp لأكبر، أو 30dp لأصغر)
                    1f);  // وزن 1
            barContainer.setLayoutParams(barContainerParams);
            barContainer.setPadding(16, 0, 16, 0);

            // الخلفية (رمادي فاتح زي الصورة، مع حواف مدورة)
            View background = new View(this);
            RelativeLayout.LayoutParams bgParams = new RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.MATCH_PARENT,
                    RelativeLayout.LayoutParams.MATCH_PARENT);
            background.setLayoutParams(bgParams);
            GradientDrawable bgDrawable = new GradientDrawable();  // <-- إضافة للـ rounded corners
            bgDrawable.setColor(Color.parseColor("#F0F0F0"));  // رمادي فاتح
            bgDrawable.setCornerRadius(15f);  // <-- هنا: غير الـ radius للمدور (15dp مثال، زد لأكتر مدور)
            background.setBackground(bgDrawable);
            barContainer.addView(background);

            // الجزء الملون الأمامي (مع حواف مدورة)
            View foreground = new View(this);
            RelativeLayout.LayoutParams fgParams = new RelativeLayout.LayoutParams(
                    (int) (percent * 3.0),  // عرض حسب النسبة (3dp لكل % تقريبًا، عدل لو عايز)
                    RelativeLayout.LayoutParams.MATCH_PARENT);
            fgParams.addRule(RelativeLayout.ALIGN_PARENT_START);  // يبدأ من اليسار
            foreground.setLayoutParams(fgParams);
            GradientDrawable fgDrawable = new GradientDrawable();  // <-- إضافة للـ rounded corners
            fgDrawable.setColor(barColors[i]);
            fgDrawable.setCornerRadius(15f);  // نفس الـ radius للتوافق
            foreground.setBackground(fgDrawable);
            barContainer.addView(foreground);

            // TextView للنسبة داخل البار الملون (غير اللون هنا)
            TextView percentTv = new TextView(this);
            percentTv.setText(percent + "%");
            percentTv.setTextSize(14);
            percentTv.setTextColor(Color.BLACK);  // <-- هنا: غير اللون (مثال: BLACK، أو Color.WHITE، أو Color.parseColor("#FF0000"))
            percentTv.setGravity(android.view.Gravity.CENTER_VERTICAL | android.view.Gravity.END);
            percentTv.setPadding(8, 0, 8, 0);
            RelativeLayout.LayoutParams percentParams = new RelativeLayout.LayoutParams(
                    RelativeLayout.LayoutParams.WRAP_CONTENT,
                    RelativeLayout.LayoutParams.WRAP_CONTENT);
            percentParams.addRule(RelativeLayout.ALIGN_PARENT_END);  // في النهاية اليمينية
            percentParams.addRule(RelativeLayout.CENTER_VERTICAL);
            percentParams.addRule(RelativeLayout.ALIGN_RIGHT, foreground.getId());  // ملتصق بالـ foreground
            if (percent < 10) {  // لو النسبة صغيرة، خليها مرئية أكتر
                percentParams.setMargins(0, 0, 4, 0);
            }
            percentTv.setLayoutParams(percentParams);
            barContainer.addView(percentTv);

            // إضافة العناصر للـ row
            row.addView(label);
            row.addView(barContainer);

            // إضافة الـ row للـ container
            confidencesContainer.addView(row);
        }
    }

    private void handleClassificationError(IOException e) {
        // TODO: Implement proper error handling
        Log.e("MainActivity", "Classification error: " + e.getMessage());
        resultTextView.setText("Classification failed");
    }

    @Override
    protected void onActivityResult(
            int requestCode, int resultCode, @Nullable Intent data
    ) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            Bitmap image = getImageFromResult(requestCode, data);
            if (image != null) {
                displayAndClassifyImage(image);
            }
        }
    }

    private Bitmap getImageFromResult(int requestCode, Intent data) {
        Bitmap image = null;
        try {
            if (requestCode == REQUEST_CAMERA) {
                // Get image from camera
                image = (Bitmap) Objects.requireNonNull(
                        data.getExtras()
                ).get("data");
            } else if (requestCode == REQUEST_GALLERY) {
                // Get image from gallery
                image = MediaStore.Images.Media.getBitmap(
                        getContentResolver(), data.getData()
                );
            }
        } catch (IOException e) {
            Log.e("MainActivity", "Error loading image: " + e.getMessage());
        }
        return image;
    }

    private void displayAndClassifyImage(Bitmap image) {
        // Create square thumbnail for display
        int dimension = Math.min(image.getWidth(), image.getHeight());
        Bitmap thumbnail = ThumbnailUtils.extractThumbnail(
                image, dimension, dimension
        );

        // Display thumbnail
        imageView.setImageBitmap(thumbnail);

        // Scale image for classification
        Bitmap scaledImage = Bitmap.createScaledBitmap(
                image, IMAGE_SIZE, IMAGE_SIZE, false
        );

        // Classify the image
        classifyImage(scaledImage);
    }
}