package com.k2fsa.sherpa.onnx.tts.engine

import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable


@Serializable
data class GithubRelease(
    @SerialName("assets")
    val assets: List<Asset> = listOf(),
//        @SerialName("assets_url")
//        val assetsUrl: String = "", // https://api.github.com/repos/jing332/frpandroid/releases/117392218/assets
//        @SerialName("author")
//        val author: Author = Author(),
//        @SerialName("body")
    val body: String = "", // > 未知CPU架构？请优先选择体积最大的APK### 更新内容：- 系统通知内容中支持显示局域网IP
//        @SerialName("created_at")
//        val createdAt: String = "", // 2023-08-16T03:02:15Z
//        @SerialName("draft")
//        val draft: Boolean = false, // false
//        @SerialName("html_url")
//        val htmlUrl: String = "", // https://github.com/jing332/frpandroid/releases/tag/1.23.081611
//        @SerialName("id")
//        val id: Int = 0, // 117392218
//        @SerialName("name")
//        val name: String = "", // 1.23.081611
//        @SerialName("node_id")
//        val nodeId: String = "", // RE_kwDOKGSbbc4G_0Na
//        @SerialName("prerelease")
//        val prerelease: Boolean = false, // false
//        @SerialName("published_at")
//        val publishedAt: String = "", // 2023-08-16T03:29:36Z
    @SerialName("tag_name")
    val tagName: String = "", // 1.23.081611
//        @SerialName("tarball_url")
//        val tarballUrl: String = "", // https://api.github.com/repos/jing332/frpandroid/tarball/1.23.081611
//        @SerialName("target_commitish")
//        val targetCommitish: String = "", // master
//        @SerialName("upload_url")
//        val uploadUrl: String = "", // https://uploads.github.com/repos/jing332/frpandroid/releases/117392218/assets{?name,label}
//        @SerialName("url")
//        val url: String = "", // https://api.github.com/repos/jing332/frpandroid/releases/117392218
//        @SerialName("zipball_url")
//        val zipballUrl: String = "" // https://api.github.com/repos/jing332/frpandroid/zipball/1.23.081611
) {
    @Serializable
    data class Asset(
        @SerialName("browser_download_url")
        val browserDownloadUrl: String = "", // https://github.com/jing332/frpandroid/releases/download/1.23.081611/AList-v1.23.081611.apk
        @SerialName("content_type")
        val contentType: String = "", // application/vnd.android.package-archive
//            @SerialName("created_at")
//            val createdAt: String = "", // 2023-08-16T03:29:37Z
//            @SerialName("download_count")
//            val downloadCount: Int = 0, // 28
//            @SerialName("id")
//            val id: Long = 0, // 121683040
//            @SerialName("label")
//            val label: String = "",
        @SerialName("name")
        val name: String = "", // AList-v1.23.081611.apk
//            @SerialName("node_id")
//            val nodeId: String = "", // RA_kwDOKGSbbc4HQLxg
        @SerialName("size")
        val size: Long = 0, // 71948726
//            @SerialName("state")
//            val state: String = "", // uploaded
//            @SerialName("updated_at")
//            val updatedAt: String = "", // 2023-08-16T03:29:39Z
//            @SerialName("uploader")
//            val uploader: Uploader = Uploader(),
//            @SerialName("url")
//            val url: String = "" // https://api.github.com/repos/jing332/frpandroid/releases/assets/121683040
    ) {
        @Serializable
        data class Uploader(
            @SerialName("avatar_url")
            val avatarUrl: String = "", // https://avatars.githubusercontent.com/in/15368?v=4
            @SerialName("events_url")
            val eventsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/events{/privacy}
            @SerialName("followers_url")
            val followersUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/followers
            @SerialName("following_url")
            val followingUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/following{/other_user}
            @SerialName("gists_url")
            val gistsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/gists{/gist_id}
            @SerialName("gravatar_id")
            val gravatarId: String = "",
            @SerialName("html_url")
            val htmlUrl: String = "", // https://github.com/apps/github-actions
            @SerialName("id")
            val id: Int = 0, // 41898282
            @SerialName("login")
            val login: String = "", // github-actions[bot]
            @SerialName("node_id")
            val nodeId: String = "", // MDM6Qm90NDE4OTgyODI=
            @SerialName("organizations_url")
            val organizationsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/orgs
            @SerialName("received_events_url")
            val receivedEventsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/received_events
            @SerialName("repos_url")
            val reposUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/repos
            @SerialName("site_admin")
            val siteAdmin: Boolean = false, // false
            @SerialName("starred_url")
            val starredUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/starred{/owner}{/repo}
            @SerialName("subscriptions_url")
            val subscriptionsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/subscriptions
            @SerialName("type")
            val type: String = "", // Bot
            @SerialName("url")
            val url: String = "" // https://api.github.com/users/github-actions%5Bbot%5D
        )
    }

    @Serializable
    data class Author(
        @SerialName("avatar_url")
        val avatarUrl: String = "", // https://avatars.githubusercontent.com/in/15368?v=4
        @SerialName("events_url")
        val eventsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/events{/privacy}
        @SerialName("followers_url")
        val followersUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/followers
        @SerialName("following_url")
        val followingUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/following{/other_user}
        @SerialName("gists_url")
        val gistsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/gists{/gist_id}
        @SerialName("gravatar_id")
        val gravatarId: String = "",
        @SerialName("html_url")
        val htmlUrl: String = "", // https://github.com/apps/github-actions
        @SerialName("id")
        val id: Int = 0, // 41898282
        @SerialName("login")
        val login: String = "", // github-actions[bot]
        @SerialName("node_id")
        val nodeId: String = "", // MDM6Qm90NDE4OTgyODI=
        @SerialName("organizations_url")
        val organizationsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/orgs
        @SerialName("received_events_url")
        val receivedEventsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/received_events
        @SerialName("repos_url")
        val reposUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/repos
        @SerialName("site_admin")
        val siteAdmin: Boolean = false, // false
        @SerialName("starred_url")
        val starredUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/starred{/owner}{/repo}
        @SerialName("subscriptions_url")
        val subscriptionsUrl: String = "", // https://api.github.com/users/github-actions%5Bbot%5D/subscriptions
        @SerialName("type")
        val type: String = "", // Bot
        @SerialName("url")
        val url: String = "" // https://api.github.com/users/github-actions%5Bbot%5D
    )
}