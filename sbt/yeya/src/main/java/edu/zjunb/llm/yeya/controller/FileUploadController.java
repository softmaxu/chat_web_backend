package edu.zjunb.llm.yeya;

import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@RestController
@RequestMapping("/api")
public class FileUploadController {

    private static final String UPLOAD_DIR = "uploads/";

    @PostMapping("/upload")
    public ResponseEntity<Map<String, Object>> uploadFiles(@RequestParam("files") MultipartFile[] files) {
        Map<String, Object> response = new HashMap<>();
        List<Map<String, Object>> fileDetails = new ArrayList<>();

        if (files.length == 0) {
            response.put("message", "No files to upload");
            return new ResponseEntity<>(response, HttpStatus.BAD_REQUEST);
        }

        try {
            // Ensure the upload directory exists
            Path uploadPath = Paths.get(UPLOAD_DIR);
            if (!Files.exists(uploadPath)) {
                Files.createDirectories(uploadPath);
            }

            for (MultipartFile file : files) {
                if (file.isEmpty()) {
                    continue;
                }

                // Save each file locally
                byte[] bytes = file.getBytes();
                Path path = Paths.get(UPLOAD_DIR + file.getOriginalFilename());
                Files.write(path, bytes);

                // Collect file details
                Map<String, Object> fileDetail = new HashMap<>();
                fileDetail.put("fileName", file.getOriginalFilename());
                fileDetail.put("fileSize", file.getSize());
                // fileDetail.put("fileUri", path.toUri().toString());
                fileDetails.add(fileDetail);
            }

            response.put("message", "Files uploaded successfully");
            response.put("fileCount", files.length);
            response.put("files", fileDetails);
            return new ResponseEntity<>(response, HttpStatus.OK);
        } catch (IOException e) {
            e.printStackTrace();
            response.put("message", "Failed to upload files");
            return new ResponseEntity<>(response, HttpStatus.INTERNAL_SERVER_ERROR);
        }
    }
}