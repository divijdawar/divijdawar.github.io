{-# LANGUAGE OverloadedStrings #-}

import           Hakyll
import           System.FilePath (takeFileName, replaceExtension)

main :: IO ()
main = hakyll $ do

    match "css/*" $ do
        route idRoute
        compile compressCssCompiler

    match "images/*" $ do
        route idRoute
        compile copyFileCompiler

    match "posts/*" $ do
        route $ customRoute $ \ident ->
            replaceExtension (drop 11 $ takeFileName $ toFilePath ident) "html"
        compile $ pandocCompiler
            >>= loadAndApplyTemplate "templates/post.html"    postCtx
            >>= loadAndApplyTemplate "templates/default.html" postCtx

    match "pages/*" $ do
        route $ gsubRoute "pages/" (const "")
        compile $ getResourceBody
            >>= loadAndApplyTemplate "templates/page.html"    defaultContext
            >>= loadAndApplyTemplate "templates/default.html" defaultContext

    match "index.html" $ do
        route idRoute
        compile $ do
            let ctx = constField "title" "Divij Dawar"
                    <> constField "bodyClass" "dark"
                    <> defaultContext
            getResourceBody
                >>= applyAsTemplate ctx
                >>= loadAndApplyTemplate "templates/default.html" ctx

    match "posts.html" $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            let ctx = listField "posts" postCtx (return posts)
                    <> constField "title" "Posts - Divij Dawar"
                    <> constField "bodyClass" "dark list"
                    <> defaultContext
            getResourceBody
                >>= applyAsTemplate ctx
                >>= loadAndApplyTemplate "templates/default.html" ctx

    create ["atom.xml"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            renderAtom feedConfig postCtx posts

    create ["sitemap.xml"] $ do
        route idRoute
        compile $ do
            posts <- recentFirst =<< loadAll "posts/*"
            pages <- loadAll "pages/*"
            let allItems = posts ++ pages
                sitemapCtx = listField "entries" defaultContext (return allItems)
                          <> defaultContext
            makeItem ""
                >>= loadAndApplyTemplate "templates/sitemap.xml" sitemapCtx

    match "templates/*" $ compile templateBodyCompiler


postCtx :: Context String
postCtx =
    dateField "date" "%B %e, %Y"
    <> defaultContext

feedConfig :: FeedConfiguration
feedConfig = FeedConfiguration
    { feedTitle       = "Divij Dawar"
    , feedDescription = "Documenting my learning notes."
    , feedAuthorName  = "Divij Dawar"
    , feedAuthorEmail = "2divijdawar@gmail.com"
    , feedRoot        = "https://divijdawar.github.io"
    }
