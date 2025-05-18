import React, { useState } from "react";

const FileUploader = ({ onFileChange, label, bannerUploaded }) => {
    const [isImageUploaded, setIsImageUploaded] = useState(false);
    const [uploadedFileName, setUploadedFileName] = useState("");

    const handleFileChange = (file) => {
        if (file) {
            setIsImageUploaded(true);
            setUploadedFileName(file.name);
        } else {
            setIsImageUploaded(false);
            setUploadedFileName("");
        }
        onFileChange(file);
    };

    return (
        <div className="file-uploader w-full">
            <label className="block text-lg font-medium text-gray-700 mb-2">
                {label}
            </label>
            <div className="relative group">
                <input
                    type="file"
                    accept="image/*"
                    onChange={(e) => handleFileChange(e.target.files[0])}
                    className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                />
                <div className={` ${ isImageUploaded ||bannerUploaded ? "bg-blue-500 " : " bg-white"} flex items-center justify-center w-full h-14 border-2 border-dashed border-gray-300 rounded-lg hover:border-indigo-500 hover:text-indigo-500 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition duration-200`}>
                    <span className={`text-sm font-medium ${ isImageUploaded || bannerUploaded ? " text-white" : " text-gray-600"} `} >
                        {isImageUploaded || bannerUploaded
                            ? `Uploaded  ${uploadedFileName}`
                            : "Click to upload or drag and drop an image"}
                    </span>
                </div>
            </div>
        </div>
    );
};

export default FileUploader;

