import React from "react";
import { Routes, Route, useLocation } from "react-router-dom";
import Home from "./pages/Home/Home";
import SuccessPage from "./pages/SuccessPage/SuccessPage";
import PreviewPage from "./pages/PreviewPage/PreviewPage";
import LogIn from "./pages/logIn/LogIn";
import SignUp from "./pages/signUp/SignUp";
import Dashboard from "./pages/Dashboard/Dashboard";
import GetStarted from "./pages/getStarted/GetStarted";
import UploadImage from "./pages/UploadImage/UploadImage";
import { NotificationProvider } from "./context/NotificationContext";
import Error from "./components/Error";
// import FabricCanvas from "./components/CanvasArea";
// import DrawingBoard from "./pages/Draw/DrawingBoard";
// import Edit from "./pages/Edit/Edit";
import Post from "./pages/Post/Post";
import Campaign from "./pages/Campaign/Campaign";
import BannerCreator from "./pages/BannerCreator/BannerCreator";


const App = () => {
  return (
    <>
      <NotificationProvider>
        <Routes>
          <Route path="/home" element={<Home />} />
          <Route path="/upload" element={<UploadImage />} />
          <Route path="/success" element={<SuccessPage />} />
          <Route path="/preview" element={<PreviewPage />} />
          <Route
            path="*"
            element={
              <Error
                errorMsg="The page you are looking for does not exist or has been removed."
                errorHeading="Page Not Found!"
              />
            }
          />

          <Route path="/Login" element={<LogIn />} />
          <Route path="/" element={<GetStarted />} />
          <Route path="/signup" element={<SignUp />} />
          <Route path="/dashboard" element={<Dashboard />} />
          {/* <Route path="/canvas" element={<FabricCanvas />} /> */}
          {/* <Route path="/draw" element={<DrawingBoard />} /> */}
          {/* <Route path="/edit" element={<Edit />} /> */}
          <Route path="/post" element={<Post />} />
          <Route path="/campaign" element={<Campaign />} />
          <Route path="/create" element={<BannerCreator />} />
        </Routes>
      </NotificationProvider>
    </>
  );
};

export default App;
