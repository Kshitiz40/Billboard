const upload_preset = import.meta.env.VITE_CLOUDINARY_UPLOAD_PRESET;
const cloud_name = import.meta.env.VITE_CLOUDINARY_CLOUD_NAME;
const api_url = import.meta.env.VITE_CLOUDINARY_URI;

export const uploadImageToCloudinary = async (file) => {
    const data = new FormData();
    data.append("file",file)
    data.append("upload_preset",upload_preset)
    data.append("cloud_name",cloud_name)
try{
    console.log(api_url)
    const res = await fetch(api_url,{
        method:"post",
        body:data,
    });
    
    const fileData = await res.json();
    return fileData.secure_url;
}
catch(error){
    console.log(error.message);
}
}